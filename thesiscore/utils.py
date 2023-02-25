"""
General utility functions. These are not specific to any deep learning framework, and therefore can be used in different contexts.
"""

# standard imports
import logging
from logging import Logger
from rich.console import Console
from rich.logging import RichHandler

__all__ = [
    "_logger",
    "_logger_format",
    "get_console",
    "get_logger",
    "login",
    "repl",
]


_logger_format: logging.Formatter = logging.Formatter(fmt="%(name)s: %(message)s")


def get_logger(name: str = None, level: int = logging.INFO):
    """
    Function to get a logger with a `RichHandler`. Sets up the logger with a custom format and a `StreamHandler`.
    ## Args:
        `name` (`str`): The name of the logger. Defaults to None.
        `level` (`int`): The level of the logger. Defaults to `logging.INFO`.
    ## Returns:
        `logging.Logger`: The logger.
    """

    logger: Logger = logging.getLogger(name)
    logger.setLevel(level=level)
    rich_handler = RichHandler(rich_tracebacks=True, level=level)
    rich_handler.setFormatter(_logger_format)
    logger.addHandler(rich_handler)
    logger.propagate = False

    return logger


def get_console(**kwargs):
    """
    Gets the rich console object.
    ## Returns:
        `Console`: Rich console object.
    """

    _console = Console(**kwargs)
    return _console


_logger: Logger = get_logger()
