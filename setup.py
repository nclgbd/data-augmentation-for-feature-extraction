"""
Source code for thesis work.
"""
import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="thesis-core",
    version="0.0.1.dev",
    packages=setuptools.find_packages(),
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
)