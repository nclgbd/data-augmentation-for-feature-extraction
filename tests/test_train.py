"""
Tests the methods in the `thesiscore.train` module.
"""

from thesiscore import repl
from thesiscore.utils import get_console

console = get_console()
repl.install(show_locals=False)

from thesiscore.config import Configuration, TorchModelConfiguration
from thesiscore.models import create_model
from thesiscore import train


class TestTrain:
    def test_create_optimizer(self, cfg: Configuration):
        """
        Tests the `create_optimizer` method.
        """
        from torch.optim import Optimizer

        model_config = cfg.model
        model: TorchModelConfiguration = create_model(**model_config)

        optimizer: Optimizer = None
        optimizer = train.create_optimizer(model, model_config)
        assert optimizer is not None

    def test_create_criterion(self, cfg: Configuration):
        """
        Tests the `create_criterion` method.
        """
        from torch import nn

        model_config = cfg.model

        criterion: nn.Module = None
        criterion = train.create_criterion(model_config)
        assert criterion is not None

    def test_create_metrics(self, cfg: Configuration):
        """
        Tests the `create_metrics` method.
        """

        model_config = cfg.model
        criterion = train.create_criterion(model_config)

        metrics: dict = None
        metrics = train.create_metrics(model_config, cfg, criterion=criterion)
        assert any(metrics)
