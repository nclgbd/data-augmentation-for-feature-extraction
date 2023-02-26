"""
Basic hydra template configurations for the `thesiscore` package.
"""
import os

from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig

# hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# torch
import torch
from torch import nn

# monai
import monai.networks.nets as mnn
import monai.transforms as monai_transforms

# thesiscore
from thesiscore.utils import get_console, get_logger

_logger = get_logger("thesiscore.config")
console = get_console()


def set_hydra_configuration(config_name: str = "config", **kwargs):
    """
    Creates and returns a hydra configuration.
    ## Args:
        `config_name` (`str`, optional): The name of the config (usually the file name without the .yaml extension). Defaults to `"config"`.
        `kwargs` (`dict`, optional): Keyword arguments for the `initialize` function.
    ## Returns:
        `DictConfig`: The hydra configuration.
    """
    console.log(f"Creating configuration: {config_name}")
    GlobalHydra.instance().clear()
    initialize(**kwargs)
    cfg: Configuration = compose(config_name=config_name)
    console.log(f"Configuration created.\n{OmegaConf.to_yaml(cfg)}")
    return cfg


@dataclass
class DatasetConfiguration:
    """
    Base configuration for a dataset.
    """

    root_dir: str = "data"
    os.makedirs(root_dir, exist_ok=True)


@dataclass
class TciaDatasetConfiguration(DatasetConfiguration):
    """
    Configuration for a TCIA dataset from monai.
    """

    collection: str = ""


@dataclass
class RunConfiguration:
    _test: bool = False
    debug: bool = False
    device: str = "cpu"
    dry_run: bool = True
    random_seed: int = 42

    # flags
    use_cv: bool = False
    use_early_stopping: bool = False
    use_lr_scheduler: bool = False
    use_mlflow: bool = False
    use_sampling: bool = False
    use_transforms: bool = False
    use_wandb: bool = False

    log_interval: int = 5
    one_vs_all: bool = False
    positive_class: int = 1
    sample_value: int = -1
    subset: list = field(default_factory=list)

    train_test_split: dict = field(default_factory=dict)


@dataclass
class TorchModelConfiguration:
    model_name: str
    input_size: field(default_factory=list)
    optimizer_name: str
    criterion_name: str
    lr_scheduler_name: str = ""
    epochs: int = 1
    model_params: DictConfig = field(default_factory=dict)
    dataloaders: DictConfig = field(default_factory=dict)
    early_stopping: DictConfig = field(default_factory=dict)
    additional_layers: list = field(default_factory=list)
    lr_schedulers: DictConfig = field(default_factory=dict)
    transforms: DictConfig = field(default_factory={"load": {}, "train": {}})


@dataclass
class WandbConfiguration:
    project: str
    entity: str
    name: str


@dataclass
class Configuration:
    dataset: DatasetConfiguration
    run: RunConfiguration
    model: TorchModelConfiguration
    metrics: DictConfig = field(default_factory=dict)
    # if run.use_wandb:
    #     wandb: WandbConfiguration = field(
    #         default_factory=WandbConfiguration(project="", entity="", name="")
    #     )
