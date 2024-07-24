import datetime
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch

import data
import config
import model


warnings.formatwarning = lambda message, category, *args, **kwargs: (
    f"{category.__name__}: {message}"
)
warnings.filterwarnings(
    "ignore",
    ".*due to potential conflicts with other packages in the ML ecosystem.*",
)
warnings.filterwarnings(
    "ignore",
    ".*The PyTorch API of nested tensors is in prototype stage.*",
)
warnings.filterwarnings(
    "ignore",
    ".*Consider increasing the value of the `num_workers` argument.*",
)
warnings.filterwarnings(
    "ignore",
    ".*nested_from_padded CUDA kernels only support fp32/fp16.*",
)


TOOL_NAME = "mfpred"

logger = logging.getLogger(TOOL_NAME)


def setup_logging(output: Optional[str], verbosity: str):
    """
    Set up the logger.

    Logging occurs to the command-line and to the given log file.

    Parameters
    ----------
    output : Optional[str]
        The provided output file name.
    verbosity : str
        The logging level to use in the console.
    """
    if output is None:
        output = (
            f"{TOOL_NAME.lower()}_"
            f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

    output = Path(output).expanduser().resolve()

    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    # Configure logging.
    logging.captureWarnings(True)
    root_logger = logging.getLogger(TOOL_NAME)
    root_logger.setLevel(logging.DEBUG)
    warnings_logger = logging.getLogger("py.warnings")

    # Formatters for file vs console.
    console_formatter = logging.Formatter("{levelname}: {message}", style="{")
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging_levels[verbosity.lower()])
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    warnings_logger.addHandler(console_handler)
    file_handler = logging.FileHandler(output.with_suffix(".log"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    warnings_logger.addHandler(file_handler)

    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(
        logging_levels[verbosity.lower()]
    )
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


setup_logging(None, "info")

pl.seed_everything(seed=config.random_seed, workers=True)
torch.set_float32_matmul_precision("medium")

# Create the training and validation data loaders.
# FIXME: DepthCharge fix needed for SpectrumDataset to be able to shuffle.
#  https://github.com/wfondrie/depthcharge/issues/62
train_dataloader = data.make_dataloader(
    config.filename_train, config.batch_size, shuffle=False
)
val_dataloader = data.make_dataloader(
    config.filename_val, config.batch_size, shuffle=False
)

os.makedirs(config.out_dir, exist_ok=True)

# Configure the model.
predictor = model.MolecularFormulaPredictor(
    config.d_model,
    config.nhead,
    config.dim_feedforward,
    config.n_layers,
    config.dropout,
    config.vocab,
    config.max_atom_cardinality,
    config.tau,
    config.lr,
)
# Train the model.
trainer = pl.Trainer(
    accelerator="auto",
    callbacks=[
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            config.out_dir,
            filename=TOOL_NAME,
            monitor="val_loss",
            mode="min",
        ),
    ],
    devices="auto",
    enable_checkpointing=True,
    enable_progress_bar=True,
    max_epochs=config.n_epochs,
    num_sanity_val_steps=0,
    precision="bf16-true",
    strategy=pl.strategies.DDPStrategy(
        find_unused_parameters=False, static_graph=True
    ),
)
trainer.fit(predictor, train_dataloader, val_dataloader)
