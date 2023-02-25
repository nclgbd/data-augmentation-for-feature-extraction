"""
Basic hydra template configurations for the `thesiscore` package.
"""

from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig

# hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# thesiscore
from thesiscore import repl
from thesiscore.utils import get_console

console = get_console()
repl.install()


def set_hydra_configuration(config_name: str = "config", **kwargs):
    """
    Creates and returns a hydra configuration.
    ## Args:
        `config_name` (`str`, optional): The name of the config (usually the file name without the .yaml extension). Defaults to `"hydracfg"`.
        `kwargs` (`dict`, optional): Keyword arguments for the `initialize` function.
    ## Returns:
        `DictConfig`: The hydra configuration.
    """
    console.log(f"Creating configuration: {config_name}")
    GlobalHydra.instance().clear()
    initialize(**kwargs)
    cfg: DictConfig = compose(config_name=config_name)
    console.log(f"Configuration created.\n{OmegaConf.to_yaml(cfg)}")
    return cfg


@dataclass
class DatasetConfiguration:
    root_dict: str


@dataclass
class TciaDatasetConfiguration(DatasetConfiguration):
    collection: str


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
    lr_schedulers: DictConfig = field(default_factory=dict)
    transforms: DictConfig = field(default_factory={"load": {}, "train": {}})


@dataclass
class Configuration:
    dataset: DatasetConfiguration
    run: RunConfiguration
    model: TorchModelConfiguration
    metrics: DictConfig = field(default_factory=dict)
