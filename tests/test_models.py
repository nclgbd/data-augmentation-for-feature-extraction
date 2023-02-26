"""
Tests the methods in the `thesiscore.models` module.
"""

from thesiscore import repl
from thesiscore.utils import get_console

console = get_console()
repl.install(show_locals=False)

from thesiscore.config import Configuration, TorchModelConfiguration
from thesiscore import models


class TestModel:
    def test_create_model(self, cfg: Configuration):
        """
        Tests the `create_model` method.
        """
        from torch import nn
        model_config: TorchModelConfiguration = cfg.model

        model: nn.Module = None

        # test using the model configuration object
        model = models.create_model(model_config)
        assert model is not None

        # test using kwargs
        model: nn.Module = None
        model = models.create_model(**model_config)
        assert model is not None
