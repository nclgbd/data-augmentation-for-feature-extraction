# common imports
from argparse import Namespace, ArgumentParser
from rich import inspect
import os

# torch imports
import torch
import torch.nn as nn

# monai imports
import monai.networks.nets as mnn

# thesiscore imports
from thesiscore.utils import get_console, get_logger
from thesiscore.config import TorchModelConfiguration

_logger = get_logger("thesiscore.models")
console = get_console()
_PATH_TO_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "model_swinvit.pt"
)


class SwinTransformerTransferModel(mnn.swin_unetr.SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.ModuleList()

    # def set_configurations(self, configurations: dict):
    #     self.configurations = configurations
    #     console.log(f"Available configurations:\n{self.configurations}\n")

    def configure_model(
        self,
        additional_layers: list,
    ):
        console.log("Configuring model...")

        # grab the name of the layer and the kwargs
        for _layer in additional_layers:
            layer_name = list(_layer.keys())[0]
            console.log(f"Adding '{layer_name}' layer...")
            layer: nn.Module = getattr(nn, layer_name)
            layer_kwargs: dict = _layer[layer_name]["kwargs"]
            self.classifier.append(layer(**layer_kwargs))

        console.log(f"Model configuration complete:\n{self}\n")

    def forward(self, x):
        output: torch.Tensor = super().forward(x)[-1]
        for layer in self.classifier:
            _logger.debug(f"Output shape:\t{output.shape}")
            _logger.debug(f"Applying layer:\t{layer}")
            output = layer(output)
        return output


def create_model(
    model_config: TorchModelConfiguration = None,
    model_name: str = None,
    model_params: dict = None,
    num_classes: int = None,
    additional_layers: list = None,
    non_blocking: bool = None,
    device: torch.device = torch.device("cpu"),
    **kwargs,
):
    """
    Creates the model for training.
    ## Args:
        `model_config` (`TorchModelConfiguration`, optional): The model configuration. Defaults to `None`.
        `model_name` (`str`, optional): The name of the model. Defaults to `None`.
        `model_params` (`dict`, optional): The parameters for the model. Defaults to `None`.
        `num_classes` (`int`, optional): The number of classes. Defaults to `None`.
        `additional_layers` (`list`, optional): Additional layers to add to the model. Defaults to `None`.
        `device` (`torch.device`, optional): The device to send the model to. Defaults to `torch.device("cpu")`.
    ## Raises:
        `NotImplementedError`: Raised if the model is not implemented.
    ## Returns:
        `nn.Module`: The model.
    """
    console.log("Creating model...")
    model_name: str = model_config.model_name if model_name is None else model_name
    model_params: dict = (
        model_config.model_params if model_params is None else model_params
    )
    num_classes: int = model_config.num_classes if num_classes is None else num_classes
    non_blocking: bool = model_config.non_blocking if non_blocking is None else non_blocking

    model: nn.Module
    if model_name == "SwinTransformer":
        # base model
        __model = mnn.SwinUNETR(**model_params)
        model_weights = torch.load(_PATH_TO_WEIGHTS, map_location=device)
        __model.load_from(model_weights)

        # save base swin transformer model just in case it's needed
        _model = __model.swinViT
        swin_trans_kwargs = {
            "in_chans": model_params["in_channels"],
            "embed_dim": _model.embed_dim,
            "window_size": _model.window_size,
            "patch_size": _model.patch_size,
            "depths": model_params["depths"],
            "num_heads": model_params["num_heads"],
        }
        model: nn.Module = SwinTransformerTransferModel(**swin_trans_kwargs)
        _logger.info(model.load_state_dict(_model.state_dict()))

        # configure model
        additional_layers = (
            model_config.additional_layers
            if additional_layers is None
            else additional_layers
        )
        model.configure_model(additional_layers=additional_layers)
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

    model = model.to(device, non_blocking=non_blocking)

    console.log("Model creation complete.\n")
    return model
