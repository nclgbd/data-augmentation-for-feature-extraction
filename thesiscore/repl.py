"""
For quick debugging and testing.
"""
from rich import pretty, traceback


def install(
    _pretty: bool = True,
    _traceback: bool = True,
    max_depth: int = 3,
    max_length: int = 7,
    show_locals: bool = True,
):
    """
    Installs the rich traceback hook and pretty install.

    ## Args:
        `max_depth` (`int`, optional): Defaults to `7`.
        `max_length` (`int`, optional): Defaults to `2`.
        `show_locals` (`bool`, optional): Defaults to `True`.
    """
    if _pretty:
        pretty.install(max_depth=max_depth, max_length=max_length)
    if _traceback:
        traceback.install(show_locals=show_locals)
