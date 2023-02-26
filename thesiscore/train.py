# torch
import torch
from torch import nn

# ignite
from ignite import metrics

from thesiscore.utils import get_console, get_logger
from thesiscore.config import RunConfiguration, TorchModelConfiguration

_logger = get_logger("thesiscore.train")
console = get_console()


def create_criterion(
    model_config: TorchModelConfiguration,
    criterion_name: str = None,
    non_blocking: bool = True,
    debug: bool = False,
    device: torch.device = torch.device("cpu"),
    **criterion_kwargs,
):
    """
    Creates a criterion for the model.
    ## Args:
        `model_config` (`TorchModelConfiguration`): The model configuration object.
        `criterion_name` (`str`, optional): The name of the criterion. Defaults to `None`.
        `debug` (`bool`, optional): Whether to perform debugging or not. Defaults to `False`.
        `device` (`torch.device`, optional): The torch device to use. Defaults to `torch.device("cpu")`.
        `non_blocking` (`bool`, optional): Whether to use `non_blocking`. Defaults to `True`.
    ## Raises:
        `NotImplementedError`: Raised when the `criterion_name` is not recognized under the `torch.nn` module.
    ## Returns:
        `nn.Module`: The criterion instance.
    """
    console.log("Creating criterion...")
    criterion: torch.nn.Module
    criterion_name = model_config.criterion_name
    non_blocking = model_config.non_blocking
    debug = model_config.debug if debug is None else False
    criterion_kwargs: dict = (
        criterion_kwargs
        if any(criterion_kwargs)
        else model_config.criterions[criterion_name]["kwargs"]
    )

    if hasattr(nn, criterion_name):
        criterion = getattr(nn, criterion_name)(**criterion_kwargs)
        criterion.to(device=device, non_blocking=non_blocking)

    else:
        raise NotImplementedError(f"Criterion {criterion_name} not recognized.")

    console.log("Criterion created.\n")
    console.log(f"Criterion summary:\n{criterion}\n")

    return criterion


def create_optimizer(
    model: nn.Module,
    model_config: TorchModelConfiguration = None,
    optimizer_name: str = None,
    **optimizer_kwargs,
):
    """
    Creates an optimizer for the model.
    ## Args:
        `model` (`nn.Module`, optional): The torch model instance.
        `model_config` (`TorchModelConfiguration`): The model configuration object.
        `optimizer_name` (`str`, optional): The name of the optimizer. Defaults to `None`.
    ## Raises:
        `NotImplementedError`: Raised when the `optimizer_name` is not recognized under the `torch.optim` module.
    ## Returns:
        `torch.optim.Optimizer`: The optimizer instance.
    """
    console.log("Creating optimizer...")
    optimizer: torch.optim.Optimizer
    optimizer_name = (
        optimizer_name if optimizer_name is not None else model_config.optimizer_name
    )
    optimizer_kwargs: dict = (
        optimizer_kwargs
        if any(optimizer_kwargs)
        else model_config.optimizers[optimizer_name]["kwargs"]
    )

    if hasattr(torch.optim, optimizer_name):
        optimizer = getattr(torch.optim, optimizer_name)(
            model.parameters(), **optimizer_kwargs
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not recognized.")

    console.log("Optimizer created.")
    console.log(f"Optimizer summary:\n{optimizer}\n")

    return optimizer


def create_metrics(
    model_config: TorchModelConfiguration,
    run_config: RunConfiguration = None,
    metric_names: list = None,
    num_classes: int = None,
    criterion: nn.Module = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Creates torch ignite metrics for the model.
    ## Args:
        `model_config` (`TorchModelConfiguration`): The model configuration object.
        `run_config` (`RunConfiguration`, optional): The run configuration object. Defaults to `None`.
        `metric_names` (`list`, optional): The torch ignite metric names. Defaults to `None`. For a list of available metrics, see [here](https://pytorch.org/ignite/metrics.html).
        `num_classes` (`int`, optional): The number of classes. Defaults to `None`.
        `device` (`torch.device`, optional): The torch device to use. Defaults to `torch.device("cpu")`.
    ## Returns:
        `dict`: A dictionary of torch ignite metrics.
    """
    console.log("Creating metrics...")
    metrics_dict = dict()
    num_classes = model_config.num_classes if num_classes is None else num_classes
    metric_names = (
        metric_names
        if metric_names is not None
        else [list(name.keys())[0] for name in run_config.metrics]
    )
    for idx, metric_name in enumerate(metric_names):
        if hasattr(metrics, metric_name):
            metric_fn_kwargs: dict = run_config.metrics[idx][metric_name]["kwargs"]
            metric_fn_kwargs = {} if metric_fn_kwargs is None else metric_fn_kwargs
            metric_fn = getattr(metrics, metric_name)
            if metric_name == "Loss":
                metric_fn_kwargs["loss_fn"] = criterion
            metrics_dict[metric_name.lower()] = (
                metric_fn(device=device, **metric_fn_kwargs)
                if any(metric_fn_kwargs)
                else metric_fn(device=device)
            )
        elif metric_name == "Predictions":
            metrics_dict["predictions"] = metrics.EpochMetric(
                device=device, compute_fn=__get_predictions
            )
        else:
            _logger.warning(f"Metric `{metric_name}` not implemented.")

    console.log("Metrics created.\n")
    _logger.debug(f"Metrics:\n{metrics_dict}\n")

    return metrics_dict


def __get_predictions(y_preds, y_true):
    y_preds = torch.argmax(y_preds, dim=1).cpu().numpy().astype(dtype=int)
    y_true = y_true.cpu().numpy()
    pred_dict = {"y_preds": y_preds, "y_true": y_true}
    return pred_dict
