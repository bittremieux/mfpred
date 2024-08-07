from typing import Dict, List

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
    vocab : Dict[str, int]
        The vocabulary of atoms and their maximum cardinalities to
        consider.
    tau : float, optional
        The temperature parameter for the Gumbel softmax.
    reg_weight : float, optional
        The weight for the adjacent class probabilities regularization
        term in the loss function.
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
        vocab: Dict[str, int],
        tau: float,
        reg_weight: float,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.tau = tau
        self.reg_weight = reg_weight
        self.lr = lr

        self.spec_encoder = SpectrumTransformerEncoder(
            d_model, n_head, d_feedforward, n_layers, dropout
        )
        self.outputs = nn.ModuleList(
            [
                nn.Linear(d_model, cardinality)
                for cardinality in vocab.values()
            ]
        )

    def forward(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> List[torch.Tensor]:
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
        logits : List[torch.Tensor]
            The predicted counts for each atom in the vocabulary as
            logits, for each spectrum in the batch.
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
        logits = [output(emb) for output in self.outputs]
        return logits

    def _compute_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute the loss for a batch of data.

        Parameters
        ----------
        batch : Dict
            A batch of training data from a `DataLoader` operating on a
            `SpectrumDataset`. The dictionary should contain at least
            the keys "mz_array", "intensity_array", "precursor_mz",
            "adduct", and "formula".

        Returns
        -------
        loss : torch.Tensor
            The loss for the batch.
        """
        # Get the target atom counts.
        targets = (
            torch.stack([batch[atom] for atom in self.vocab], dim=0)
            .to(torch.float)
            .to(self.device)
        )

        # Predict the atom counts.
        logits = self(
            batch["mz_array"],
            batch["intensity_array"],
            precursor_mz=batch["precursor_mz"],
            adduct=batch["adduct"],
        )

        # Compute the loss.
        mse_loss = torch.zeros(1, device=self.device)
        diff_penalty = torch.zeros(1, device=self.device)

        for output, target in zip(logits, targets):
            # Apply the Gumbel softmax.
            gumbel_probs = F.gumbel_softmax(output, tau=self.tau, hard=False)
            # Convert the probabilities into a weighted sum, which
            # represents the expected value of the atomic counts.
            predictions = gumbel_probs @ torch.arange(
                gumbel_probs.size(1), device=self.device, dtype=torch.bfloat16
            )
            # Compute the mean squared error, normalized by the maximum
            # cardinality of the atom.
            mse_loss += torch.sum(
                ((predictions - target)) ** 2
            )
            # Penalize large differences between adjacent probabilities.
            diff_penalty += torch.sum(
                (gumbel_probs[:, 1:] - gumbel_probs[:, :-1]) ** 2
            )
        # Normalize the losses by the batch size.
        batch_size = batch["mz_array"].shape[0]
        mse_loss /= batch_size
        diff_penalty /= batch_size
        # Combine the MSE loss and the regularization penalty.
        loss = mse_loss + self.reg_weight * diff_penalty
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        Compute the loss for a batch of training data.

        Parameters
        ----------
        batch : Dict
            A batch of training data from a `DataLoader` operating on a
            `SpectrumDataset`. The dictionary should contain at least
            the keys "mz_array", "intensity_array", "precursor_mz",
            "adduct", and "formula".
        batch_idx : int
            The index of the batch.

        Returns
        -------
        loss : torch.Tensor
            The loss for the batch.
        """
        loss = self._compute_loss(batch)
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

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        Compute the loss for a batch of validation data.

        Parameters
        ----------
        batch : Dict
            A batch of training data from a `DataLoader` operating on a
            `SpectrumDataset`. The dictionary should contain at least
            the keys "mz_array", "intensity_array", "precursor_mz",
            "adduct", and "formula".
        batch_idx : int
            The index of the batch.

        Returns
        -------
        loss : torch.Tensor
            The loss for the batch.
        """
        loss = self._compute_loss(batch)
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

    def predict_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        Make predictions for a batch of spectra.

        Parameters
        ----------
        batch : Dict
            A batch of spectra from a `DataLoader` operating on a
            `SpectrumDataset`. The dictionary should contain at least
            the keys "mz_array", "intensity_array", "precursor_mz",
            "adduct", and "formula".
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, vocab_size)
            The predicted atom counts for each spectrum in the batch.
        """
        logits = self(
            batch["mz_array"],
            batch["intensity_array"],
            precursor_mz=batch["precursor_mz"],
            adduct=batch["adduct"],
        )
        # Alternatively, the Gumbel softmax (with hard=False) could be
        # used to understand the uncertainty in the predictions.
        predictions = torch.stack(
            [torch.argmax(output, dim=1) for output in logits], dim=1
        )
        return predictions

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Currently, this is a simple Adam optimizer.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
