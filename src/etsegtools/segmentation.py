import numpy as np
import mrcfile
import os
from typing import List, Tuple

from skimage import io

import qvox.sampling
import qvox.utils
from qvox.morphology import grow, shrink, gaussian_smooth

class Segmentation:
    def __init__(self, data: np.ndarray, label_dict: dict[str, int], pixsize: float = 1) -> None:
        """
        Initializes a Segmentation object with a 3D numpy array of integers and a dictionary which connects the ID values to their semantic labels.
        
        Parameters
        ----------
        data : numpy.ndarray
            3D numpy array of integers representing the segmentation.
        label_dict : dict: str->int
            Dictionary which connects the ID values to their semantic labels. Label->ID value mapping
        pixsize : float, optional
            The pixel size of the segmentation, by default 1
        """
        self.data = data
        self.label_dict = label_dict
        self.pixsize = pixsize
    
    def get_binary_array(self, label: str) -> np.ndarray:
        """
        Returns a binary array showing values corresponding to a given string label.
        
        Parameters
        ----------
        label : str
            The string label to generate the binary array for.
        
        Returns
        -------
        numpy.ndarray
            Binary array showing values corresponding to the given string label.
        
        Raises
        ------
        AssertionError
            If the label is not found among the labels for this segmentation.
        """
        # Get the ID value corresponding to the given label
        assert(label in self.label_dict.keys(), f"{label} not found among labels for this segmentation. The labels are {self.label_dict.keys()}")
        # Create a binary array with True values where the data array matches the label ID and False elsewhere
        binary_array = (self.data == self.label_dict[label]).astype(int)
        
        return binary_array
    
    def get_array_from_id(self, labelid: int) -> np.ndarray:
        """
        Returns a binary array showing values corresponding to a given int id.
        
        Parameters
        ----------
        labelid : int
            The integer id to generate the binary array for.
        
        Returns
        -------
        numpy.ndarray
            Binary array showing values corresponding to the given string label.
        """
        # Create a binary array with 1 values where the data array matches the label ID and False elsewhere
        binary_array = (self.data == labelid).astype(int)
        
        return binary_array
    
    def write_mrcfile(self, file_path: str) -> None:
        """
        Writes the segmentation data to an MRC file with the given file path. The label dictionary is stored in the extended header.
        
        Parameters
        ----------
        file_path : str
            The file path to write the MRC file to.
        """
        with mrcfile.new(file_path, overwrite=True) as mrc:
            mrc.voxel_size = self.pixsize
            mrc.set_data(self.data.astype(float))
            mrc.set_extended_header(self.label_dict.keys())
    
    def write_dragonfly(self, folder_path: str, also_write_mrc: bool = False) -> None:
        """
        Generates a folder with a folder for each label containing 2D tif each showing a single Z slice of the binary array of that label.
        
        Parameters
        ----------
        folder_path : str
            The path to the folder to generate.
        also_write_mrc : bool, optional
            Whether to also write an MRC file containing the segmentation data, by default False
        """
        os.makedirs(folder_path, exist_ok=True)
        if also_write_mrc:
            self.write_mrcfile(os.path.join(folder_path, "combined.mrc"))
        for label in self.label_dict.keys():
            binary_array = self.get_binary_array(label)
            binary_array = binary_array.astype(np.uint8)
            io.imsave(os.path.join(folder_path, f"{label}.tiff"), binary_array*255, plugin="tifffile", check_contrast=False)
    
    def morphological_smooth(self, iter: int = 2) -> None:
        """
        Smooths the data by performing successive grow and shrink operations
        
        Parameters
        ----------
        iterations : int
            The the number of iterations to grow then shrink. Larger numbers smooth better but can merge adjacent sections, which is undesirable.
        """
        new_data = grow(self.data, num_iterations=iter)
        new_data = shrink(new_data, num_iterations=iter)
        self.data = new_data

    def gaussian_smooth(self, sigma: float = 1.0, thresh: float = 0.1) -> None:
        """
        Smooths the data by performing successive grow and shrink operations
        
        Parameters
        ----------
        sigma : int
            The sigma by which to perform gaussian filtering
        thresh : float
            The threshold by which to re-binarize.
        """
        new_data = gaussian_smooth(self.data, sigma=sigma, threshold=thresh)
        self.data = new_data

    def rescale(self, new_pixsize: float, thresh: float = 0.5) -> None:
        """
        Rescales the data array to the desired pixel spacing using qvox.sampling.rescale
        
        Parameters
        ----------
        new_pixsize : float
            The desired pixel spacing for the rescaled data array.
        thresh : float
            The threshold by which to re-binarize.

        """
        self.data = qvox.sampling.rescale(self.data, self.pixsize, new_pixsize, threshold = thresh)
        self.pixsize = new_pixsize

    


def read_dragonfly(folder_name: str, labels: List[str], pixsize: float = 1) -> Segmentation:
    """
    Generates a Segmentation object by combining individual labels into a single quantized numpy 3d array with integer values corresponding to a new label_dict.
    
    Parameters
    ----------
    folder_name : str
        The name of the folder containing the label folders.
    labels : List[str]
        The list of labels to generate the segmentation for.
    pixsize : float, optional
        The pixel size of the segmentation, by default 1
    
    Returns
    -------
    Segmentation
        The Segmentation object generated from the individual labels.
    """

    # Initialize an empty dictionary to store the label ID values
    label_dict = {}
    
    # Initialize an empty list to store the individual label arrays
    label_arrays = []
    
    # Loop through each label and load the corresponding tifs into a 3D array
    for label in labels:
        # Load the 3D tiff into an array
        label_array = io.imread(os.path.join(folder_name, f"{label}.tiff"))
    
        # Add the label ID value to the label_dict
        label_dict[label] = len(label_dict)+1
        
        # Quantize the label array and add it to the list of label arrays
        label_arrays.append((label_array > 0).astype(int))
    
    # Combine the individual label arrays into a single 3D array
    data = qvox.utils.combine_binary_arrays(label_arrays)
    # Create a Segmentation object with the combined 3D array and the label_dict
    segmentation = Segmentation(data, label_dict, pixsize=pixsize)
    
    return segmentation

def read_mrcfile(file_path: str, label_list: List[str] = None) -> Segmentation:
    """
    Reads an MRC file containing a single 3D array of integer values as well as a list of labels corresponding to integer values, and returns a Segmentation with that data and label list.
    
    Parameters
    ----------
    file_path : str
        The file path to the MRC file to read.
    label_list : List[str], optional
        The list of labels corresponding to integer values, by default None.
    
    Returns
    -------
    Segmentation
        The Segmentation object generated from the MRC file.
    """
    with mrcfile.open(file_path) as mrc:
        # Get the data array from the MRC file
        data = mrc.data
        pixsize = mrc.voxel_size.x.astype(float)
        # Get the label dictionary from the extended header of the MRC file
        if label_list is None:
            label_list = mrc.extended_header
        label_dict = {val: i for i,val in enumerate(label_list)}
        
        # Create a Segmentation object with the data array and label dictionary
        segmentation = Segmentation(data, label_dict)
        
        return segmentation

