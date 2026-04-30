"""Setup script for fg-strategy-modelling-core package."""

from setuptools import setup, find_packages

setup(
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
)
