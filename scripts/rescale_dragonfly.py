import os
import click
import etsegtools
import numpy as np

@click.command()
@click.option('--input_folder', type=click.Path(exists=True), help='input folder path', required=True)
@click.option('--orig_pixel_size', type=float, help='original pixel size', required=True)
@click.option('--output_folder', type=click.Path(), help='output folder path', required=True)
@click.option('--output_pixel_size', type=float, help='output pixel size', required=True)
def load_rescale_write(input_folder, orig_pixel_size, output_folder, output_pixel_size):
    """
    Load a dragonfly folder, rescale the segmentation, and save as a new dragonfly folder.

    read the list of folders inside the input folder to determine the list of labels
    """
    # List all filenames inside the input folder
    labels = [os.path.splitext(file)[0] for file in os.listdir(input_folder) if file.endswith('.tiff') or file.endswith('.tif')]
    seg = etsegtools.read_dragonfly(input_folder, labels=labels, pixsize=orig_pixel_size)
    # Rescale from the original pixel size to the output pixel size
    seg.rescale(output_pixel_size, thresh = 0.3)
    seg.morphological_smooth(iter=2)
    seg.gaussian_smooth(sigma=2.5, thresh = 0.5)
    # Output a new dragonfly folder
    seg.write_mrcfile(os.path.join(output_folder, "totalseg.mrc"))
    seg.write_dragonfly(output_folder)


if __name__ == '__main__':
    load_rescale_write()