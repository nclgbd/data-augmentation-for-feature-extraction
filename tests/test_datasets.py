"""
Tests the methods in the `thesiscore.datasets` module.
"""

from thesiscore import repl
from thesiscore.utils import get_console

console = get_console()
repl.install(show_locals=False)

from thesiscore.config import Configuration, TorchModelConfiguration
from thesiscore import datasets


class TestDatasets:
    def test_create_transforms(self, cfg: Configuration):
        """
        Tests the `create_transforms` method.
        """
        model_config: TorchModelConfiguration = cfg.model

        # check length of transforms when not using transforms
        transforms = datasets.create_transforms(model_config, use_transforms=False)
        load_transforms = model_config.transforms["load"]
        assert len(transforms.transforms) == len(load_transforms) + 1

        # check length of transforms when using transforms
        all_transforms = load_transforms + model_config.transforms["train"]
        transforms = datasets.create_transforms(model_config, use_transforms=True)
        assert len(transforms.transforms) == len(all_transforms) + 1
