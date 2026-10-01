"""Setup script for fg_core package."""

from setuptools import setup, find_packages

setup(
    packages=find_packages(include=["fg_core", "fg_core.*"]),
    package_dir={"": "."},
)
