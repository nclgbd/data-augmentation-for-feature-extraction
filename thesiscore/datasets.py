# monai
import monai.transforms as monai_transforms

# thesiscore
from thesiscore.config import TorchModelConfiguration
from thesiscore.utils import get_console, get_logger

_logger = get_logger("thesiscore.datasets")
console = get_console()


def __get_monai_transforms(
    transforms: list,
):

    ret_transforms = []
    for _transform in transforms:
        transform_name = list(_transform.keys())[0]
        if hasattr(monai_transforms, transform_name):
            transform = getattr(monai_transforms, transform_name)
            transform_kwargs = _transform[transform_name]["kwargs"]
            if transform_kwargs is not None:
                ret_transforms.append(transform(**transform_kwargs))
            else:
                ret_transforms.append(transform())
        else:
            _logger.warning("Transform {} not found".format(transform_name))

    # always convert to tensor at the end
    ret_transforms.append(monai_transforms.ToTensor())
    return monai_transforms.Compose(ret_transforms)


def create_transforms(
    model_config: TorchModelConfiguration = None,
    use_transforms: bool = False,
    transform_dicts: dict = None,
    **kwargs,
):
    """
    Get transforms for the model based on the model configuration.
    ## Args:
        `model_config` (`TorchModelConfiguration`, optional): The model configuration. Defaults to `None`.
        `use_transforms` (`bool`, optional): Whether or not to use the transforms. Defaults to `False`.
        `transform_dicts` (`dict`, optional): The dictionary of transforms to use. Defaults to `None`.
    ## Returns:
        `torchvision.transforms.Compose`: The transforms for the model in the form of a `torchvision.transforms.Compose`
        object.
    """
    console.log("Creating transforms...")
    transform_dicts: dict = (
        model_config.transforms if transform_dicts is None else transform_dicts
    )

    # transforms specific to loading the data. These are always used
    transforms: list = transform_dicts["load"]

    # If we're using transforms, we need to load the training dictionaries as well
    if use_transforms:
        # Grab train transforms from the dictionary
        transforms += transform_dicts["train"]

    _transform_names = [list(transform.keys())[0] for transform in transforms]

    ret_transforms = __get_monai_transforms(transforms)
    console.log("Transform creation complete:\t{}\n".format(_transform_names))

    return ret_transforms
