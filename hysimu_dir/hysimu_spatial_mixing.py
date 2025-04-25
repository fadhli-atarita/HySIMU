# ======================================================================= #
"""
hysimu_spatial_mixing
------
A function to perform spatial mixing parallelly with joblib to downsample
the surface reflectance datacube to sensor resolution, using a previously
computed net PSF filter.
"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
from scipy.signal import convolve2d
from joblib import Parallel, delayed
import numpy as np
from skimage.transform import resize


# ======================================================================= #
# SLICE CONVOLUTION FUNCTION
# ======================================================================= #

def slice_convolution(
    datacube_slice,
    num_col,
    num_row,
    net_psf,
    mix_order,
    cores
):
    """
    Perform convolution on wavelength-sliced datacube (axis=-1) with
    net psf

    Parameters:
        - datacube_slice (array, float): a slice surface reflectance
                                            datacube (on 3rd dimension)
        - num_col (int): 2nd dimension size of the field
        - num_row (int): 1nd dimension size of the field
        - net_psf (array, float): previously computed net psf filter
        - mix_order (int): spatial mixing order to downsample.
                            1=linear and so on
        - cores (int): how many processing cores to run joblib on

    Returns:
        - mix_datacube_slice (array, float): a slice of spatially mixed
                                                datacube
    """

    # Perform slice convolution
    rows = datacube_slice.shape[0]

    # Run joblib
    datacube_slice_conv = Parallel(n_jobs=cores)(
        delayed(
            lambda row: convolve2d(
                datacube_slice[row:row + 1, :], net_psf,
                mode='same', boundary='symm')[0]
        )
        (row) for row in range(rows)
    )

    # Change slice to an array
    datacube_slice_conv_ar = np.asarray(datacube_slice_conv)

    # Spatially mix the convolved datacube slice
    mix_datacube_slice = resize(
        datacube_slice_conv_ar,
        (num_row, num_col),
        order=mix_order,
        mode='reflect',
        anti_aliasing=False  # no need for anti-aliasing after using PSF
    )

    return mix_datacube_slice


# ======================================================================= #
# MAIN FUNCTION
# ======================================================================= #
def main(
    datacube,
    num_col,
    num_row,
    new_datacube,
    net_psf,
    mix_order,
    cores,
):
    """
    Perform parellel joblib process on slice convolution function

    Parameters:
        - datacube (array, float): surface reflectance datacube
        - num_col (int): 2nd dimension size of the field
        - num_row (int): 1nd dimension size of the field
        - new_datacube (array, float): empty output datacube at sensor
                                        resolution
        - net_psf (array, float): previously computed net psf filter
        - mix_order (int): spatial mixing order to downsample.
                            1=linear and so on
        - cores (int): how many processing cores to run joblib on

    Returns:
        - new_datacube (array, float): spatially-downsampled output datacube
    """

    # Get 3rd dimension index to cut slices
    k_index = datacube.shape[2]

    for k in range(k_index):
        # Loop through the 3rd dimension and convolve each slice with
        # net PSF
        new_datacube[:, :, k] = slice_convolution(
            datacube_slice=datacube[:, :, k],
            num_col=num_col,
            num_row=num_row,
            net_psf=net_psf,
            cores=cores,
            mix_order=mix_order
        )

    return new_datacube


# ======================================================================= #
# INITIALIZE MAIN FUNCTION
# ======================================================================= #
if __name__ == "__main__":
    main()


# ======================================================================= #
# ======================================================================= #
