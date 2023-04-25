"""Tools for manipulating multilabel cryoET segmentations"""
from importlib.metadata import PackageNotFoundError, version
from segmentation import Segmentation, read_dragonfly, read_mrcfile

try:
    __version__ = version("ETSegTools")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
