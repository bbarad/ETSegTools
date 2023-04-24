"""Tools for manipulating multilabel cryoET segmentations"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ETSegTools")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
