import pytest
from thesiscore.config import set_hydra_configuration


@pytest.fixture
def config_path():
    return "../config"


@pytest.fixture
def config_name():
    return "config"


@pytest.fixture
def cfg(config_path, config_name):
    cfg = set_hydra_configuration(config_name=config_name, config_path=config_path)
    return cfg
