from typing import Dict, Sequence

import depthcharge as dc
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class SpectrumTransformerEncoder(dc.transformers.SpectrumTransformerEncoder):

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_feedforward: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__(d_model, n_head, d_feedforward, n_layers, dropout)
        self.precursor_mz_encoder = dc.encoders.FloatEncoder(d_model)
        # TODO: Add aduct encoder.
        self.fc_combined = nn.Linear(2 * d_model, d_model)

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Use the global token hook mechanism to encode additional
        information.

        The global token hook processes:
        - precursor_mz: The precursor m/z values of the spectra.
        - adduct: The adducts of the spectra.

        Parameters
        ----------
        mz_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded m/z dimension for a batch of MS/MS spectra.
        intensity_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded intensity dimension for a batch of MS/MS
            spectra.
        args : torch.Tensor
            Additional data passed with the batch.
        kwargs : dict
            Additional data passed with the batch. It should contain at
            least the key "precursor_mz".

        Returns
        -------
        torch.Tensor of shape (n_spectra, d_model)
            The precursor representations.
        """
        precursor_mz_emb = self.precursor_mz_encoder(
            kwargs["precursor_mz"].type_as(mz_array)[None, :],
        )
        # TODO: Add adduct embedding.
        adduct_emb = torch.zeros_like(precursor_mz_emb)
        # Combine the different precursor-level embeddings.
        combined_emb = torch.cat((precursor_mz_emb, adduct_emb), dim=-1)
        return self.fc_combined(combined_emb).squeeze()


class MolecularFormulaPredictor(pl.LightningModule):
    """
    A model to predict molecular formulae from MS/MS spectra.

    The model consists of a transformed-based spectrum encoder, based on
    Yilmaz et al. (2022) [1]_, followed by a linear layer to predict
    counts of individual atoms constituting the molecular formula.
    Atom counts are predicted as an ordinal regression task using the
    Gumbel softmax.

    Parameters
    ----------
    d_model : int
        The latent dimensionality used by the transformer spectrum
        encoder.
    n_head : int
        The number of attention heads in each layer. `d_model` must be
        divisible by `n_head`.
    d_feedforward : int
        The dimensionality of the fully connected layers in the
        transformer encoder.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    vocab : Sequence[str]
        The vocabulary of atoms to consider.
    max_atom_cardinality : int
        The maximum atom cardinality to consider.
    tau : float, optional
        The temperature parameter for the Gumbel softmax.
    lr : float
        The learning rate for training used by the Adam optimizer.

    References
    ----------
    .. [1] Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble,
    W. S. De Novo Mass Spectrometry Peptide Sequencing With a
    Transformer Model. in Proceedings of the 39th International
    Conference on Machine Learning - ICML '22 vol. 162 25514–25522
    (PMLR, Baltimore, MD, USA, 2022).
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_feedforward: int,
        n_layers: int,
        dropout: float,
        vocab: Sequence[str],
        max_atom_cardinality: int,
        tau: float,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_atom_cardinality = max_atom_cardinality
        self.tau = tau
        self.lr = lr

        self.spec_encoder = SpectrumTransformerEncoder(
            d_model, n_head, d_feedforward, n_layers, dropout
        )
        self.fc = nn.Linear(
            d_model, self.vocab_size * self.max_atom_cardinality
        )

    def forward(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the atom counts from an MS/MS spectrum.

        Parameters
        ----------
        mz_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded m/z dimension for a batch of MS/MS spectra.
        intensity_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded intensity dimension for a batch of MS/MS
            spectra.
        args : torch.Tensor
            Additional data used by the spectrum encoder, as implemented
            using the `global_token_hook()` in DepthCharge.
        mask : torch.Tensor, optional
            Passed to `torch.nn.TransformerEncoder.forward()`. The mask
            for the sequence.
        kwargs : dict
            Additional data fields used by the spectrum encoder, as
            implemented using the `global_token_hook()` in DepthCharge.

        Returns
        -------
        logits : torch.Tensor of shape (n_spectra, n_atoms)
            The predicted atom counts for each spectrum in the batch.
        """
        # Encode the spectrum.
        emb, _ = self.spec_encoder(
            mz_array, intensity_array, *args, mask=mask, **kwargs
        )
        # The encoder returns the embeddings for the full spectrum,
        # followed by the embeddings for each peak. Only retain the
        # embeddings for the spectrum itself.
        emb = emb[:, 0, :]
        # Predict the atom counts from the spectrum embeddings.
        logits = self.fc(emb).view(
            -1, self.vocab_size, self.max_atom_cardinality
        )
        return logits

    def step(self, batch: Dict) -> torch.Tensor:
        """
        Compute the loss for a batch of training data.

        Parameters
        ----------
        batch : Dict
            A batch of training data from a `DataLoader` operating on a
            `SpectrumDataset`. The dictionary should contain at least
            the keys "mz_array", "intensity_array", "precursor_mz", and
            "formula".

        Returns
        -------
        loss : torch.Tensor
            The loss for the batch.
        """
        # Get the target atom counts.
        targets = (
            torch.stack([batch[atom] for atom in self.vocab], dim=1)
            .to(torch.long)    # Ensure targets are long for cross-entropy.
            .to(self.spec_encoder.device)
        )
        # Predict the atom counts.
        logits = self(
            batch["mz_array"],
            batch["intensity_array"],
            precursor_mz=batch["precursor_mz"],
        )

        # Apply the Gumbel softmax.
        logits = F.gumbel_softmax(logits, tau=self.tau, hard=True)

        # Compute the loss.
        # Logits are shape (batch_size, vocab_size, max_atom_cardinality).
        # Targets are shape (batch_size, vocab_size).
        loss = torch.zeros(1, device=self.spec_encoder.device)
        for i in range(self.vocab_size):
            loss += F.cross_entropy(logits[:, i, :], targets[:, i])
        loss /= self.vocab_size  # Normalize by vocab_size.

        return loss

    def training_step(
        self,
        batch,
        batch_idx: int,
    ):
        """
        Compute the loss for a batch of training data.

        Parameters
        ----------
        batch : Dict
            A batch of training data from a `DataLoader` operating on a
            `SpectrumDataset`. The dictionary should contain at least
            the keys "mz_array", "intensity_array", "precursor_mz", and
            "formula".
        batch_idx : int
            The index of the batch.

        Returns
        -------
        loss : torch.Tensor
            The loss for the batch.
        """
        loss = self.step(batch)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["mz_array"].shape[0],
        )
        return loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ):
        """
        Compute the loss for a batch of validation data.

        Parameters
        ----------
        batch : Dict
            A batch of validation data from a `DataLoader` operating on
             a`SpectrumDataset`. The dictionary should contain at least
            the keys "mz_array", "intensity_array", "precursor_mz", and
            "formula".
        batch_idx : int
            The index of the batch.

        Returns
        -------
        loss : torch.Tensor
            The loss for the batch.
        """
        loss = self.step(batch)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["mz_array"].shape[0],
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Currently, this is a simple Adam optimizer.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
