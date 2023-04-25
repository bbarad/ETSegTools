import numpy as np
import mrcfile
import os
from skimage import io

class Segmentation:
    def __init__(self, data: np.ndarray, label_dict: dict[str, int], pixsize: float = 1):
        """
        Initializes a Segmentation object with a 3D numpy array of integers and a dictionary which connects the ID values to their semantic labels.
        
        Args:
        data (numpy.ndarray): 3D numpy array of integers representing the segmentation.
        label_dict (dict: str->int): Dictionary which connects the ID values to their semantic labels. Label->ID value mapping
        """
        self.data = data
        self.label_dict = label_dict
        self.pixsize = pix_size
    
    def get_binary_array(self, label: str) -> np.ndarray:
        """
        Returns a binary array showing values corresponding to a given string label.
        
        Args:
        label (str): The string label to generate the binary array for.
        
        Returns:
        numpy.ndarray: Binary array showing values corresponding to the given string label.
        """
        # Get the ID value corresponding to the given label
        assert(label in self.label_dict.keys(), f"{label} not found among labels for this segmentation. The labels are {self.label_dict.keys()}")

        # Create a binary array with True values where the data array matches the label ID and False elsewhere
        binary_array = (self.data == self.label_dict[label]).astype(int)
        
        return binary_array
    
    def get_array_from_id(self, labelid: int) -> np.ndarray:
        """
        Returns a binary array showing values corresponding to a given int id.
        
        Args:
        labelid (int): The integer id to generate the binary array for.
        
        Returns:
        numpy.ndarray: Binary array showing values corresponding to the given string label.
        """
        # Create a binary array with 1 values where the data array matches the label ID and False elsewhere
        binary_array = (self.data == labelid).astype(int)
        
        return binary_array
    
    def write_mrcfile(self, file_path: str):
        """
        Writes the segmentation data to an MRC file with the given file path. The label dictionary is stored in the extended header.
        
        Args:
        file_path (str): The file path to write the MRC file to.
        """
        with mrcfile.new(file_path, overwrite=True) as mrc:
            mrc.voxel_size = self.pixsize
            mrc.set_data(self.data)
            mrc.set_extended_header(self.label_dict.keys())
    
    def write_dragonfly_tifs(self, folder_path: str, also_write_mrc: bool = False):
        """
        Generates a folder with a folder for each label containing 2D tif each showing a single Z slice of the binary array of that label.
        
        Args:
        folder_path (str): The path to the folder to generate.
        """
        os.makedirs(folder_path, exist_ok=True)
        if also_write_mrc:
            self.write_mrcfile(os.path.join(folder_path, "combined.mrc"))
        for label in self.label_dict.keys():
            binary_array = self.get_binary_array(label)
            label_folder_path = os.path.join(folder_path, label)
            os.makedirs(label_folder_path, exist_ok=True)
            for z in range(binary_array.shape[2]):
                io.imsave(os.path.join(label_folder_path, f"{label}{z:03d}.tif"), binary_array[:, :, z], resolution=(1/self.pixsize, 1/self.pixsize))
    
    


def read_dragonfly(folder_name: str, labels: List[str], pixsize: float = 1) -> Segmentation:
    """
    Generates a Segmentation object by combining individual labels into a single quantized numpy 3d array with integer values corresponding to a new label_dict.
    
    Args:
    folder_name (str): The name of the folder containing the label folders.
    labels (List[str]): The list of labels to generate the segmentation for.
    
    Returns:
    Segmentation: The Segmentation object generated from the individual labels.
    """
    # Initialize an empty dictionary to store the label ID values
    label_dict = {}
    
    # Initialize an empty list to store the individual label arrays
    label_arrays = []
    
    # Loop through each label and load the corresponding tifs into a 3D array
    for label in labels:
        # Load the tifs for the current label into a list
        tif_list = []
        for file_name in os.listdir(os.path.join(folder_name, label)):
            if file_name.endswith(".tif") and file_name.startswith(label):
                tif_list.append(io.imread(os.path.join(folder_name, label, file_name)))
        
        # Combine the individual tifs into a 3D array based on slice number
        label_array = np.stack(tif_list, axis=2)
        
        # Add the label ID value to the label_dict
        label_dict[label] = len(label_dict)
        
        # Quantize the label array and add it to the list of label arrays
        label_arrays.append((label_array > 0).astype(int) * label_dict[label])
    
    # Combine the individual label arrays into a single 3D array
    data = np.sum(label_arrays, axis=0)
    
    # Create a Segmentation object with the combined 3D array and the label_dict
    segmentation = Segmentation(data, label_dict)
    
    return segmentation

def read_mrcfile(file_path: str, label_list: List[str] = None) -> Segmentation:
    """
    Reads an MRC file containing a single 3D array of integer values as well as a list of labels corresponding to integer values, and returns a Segmentation with that data and label list.
    
    Args:
    file_path (str): The file path to the MRC file to read.
    
    Returns:
    Segmentation: The Segmentation object generated from the MRC file.
    """
    with mrcfile.open(file_path) as mrc:
        # Get the data array from the MRC file
        data = mrc.data
        pixsize = mrc.voxel_size.x.astype(float)
        # Get the label dictionary from the extended header of the MRC file
        if label_list == None:
            label_list = mrc.extended_header
        label_dict = {val: i for i,val in enumerate(label_list)}
        
        # Create a Segmentation object with the data array and label dictionary
        segmentation = Segmentation(data, label_dict)
        
        return segmentation


