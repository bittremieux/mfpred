import functools
import re
from typing import Dict, Sequence

import depthcharge as dc
import pyarrow as pa
import torch

import config


# Regular expression to match atoms and their counts.
token_pattern = re.compile(r"([A-Z][a-z]*)(\d*)")


@functools.lru_cache(maxsize=None)
def split_formula(formula: str, vocab: Sequence[str]) -> Dict[str, int]:
    """
    Split a molecular formula into atom counts.

    Parameters
    ----------
    formula : str
        The molecular formula.
    vocab : Sequence[str]
        The vocabulary of atoms to consider.

    Returns
    -------
    Dict[str, int]
        A dictionary of counts for all known atoms.

    Raises
    ------
    ValueError
        If an unknown atom is encountered.
    """
    atom_counts = {atom: 0 for atom in vocab}
    for atom, count in re.findall(token_pattern, formula):
        if atom in vocab:
            atom_counts[atom] = int(count) if count else 1
        else:
            raise ValueError(f"Unknown atom: {atom}")

    return atom_counts


def make_dataloader(
    filename: str, batch_size: int, shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a `DataLoader` from the given spectrum input file.

    Parameters
    ----------
    filename : str
        The input file from which to read the spectra.
    batch_size : int
        The batch size.
    shuffle : bool, optional
        Whether to shuffle the data.

    Returns
    -------
    torch.utils.data.DataLoader
        The `DataLoader`.
    """
    dataset = dc.data.SpectrumDataset(
        filename,
        batch_size=batch_size,
        parse_kwargs=dict(
            custom_fields=[
                # FIXME: For easy debugging purposes.
                dc.data.CustomField(
                    "formula",
                    lambda x: x["params"]["precursor_formula"].strip(),
                    pa.string(),
                ),
                dc.data.CustomField(
                    "adduct",
                    lambda x: x["params"]["adduct"].strip(),
                    pa.string(),
                ),
                *[
                    dc.data.CustomField(
                        atom,
                        lambda x, atom=atom: split_formula(
                            x["params"]["precursor_formula"].strip(),
                            config.vocab,
                        )[atom],
                        pa.int8(),
                    )
                    for atom in config.vocab
                ],
            ],
            preprocessing_fn=[
                dc.data.preprocessing.set_mz_range(config.min_mz),
                dc.data.preprocessing.remove_precursor_peak(
                    config.remove_precursor_tol_mass,
                    config.remove_precursor_tol_mode,
                ),
                dc.data.preprocessing.filter_intensity(
                    config.min_intensity, config.max_num_peaks
                ),
                dc.data.preprocessing.scale_intensity(config.scaling),
                dc.data.preprocessing.scale_to_unit_norm,
            ],
        ),
    )
    return torch.utils.data.DataLoader(
        dataset,
        # FIXME: https://github.com/wfondrie/depthcharge/issues/62
        batch_size=None,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
