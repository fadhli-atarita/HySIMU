# ======================================================================= #
"""
hysimu_psf_simplified
A function to compute a simplified PSF for hyperspectral imaging.
A simplified PSF filter based on a gaussian distribution to simulate
an airy disk convolved with a rectangular response function
image motion and electronic response are ignored.

Calculation of the PSFs follows formulas provided in:
    [Schowengerdt, R. A. (2007). Remote sensing, models, and
    methods for image processing (3rd ed). Academic Press. p85-91]
"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
from scipy.signal import convolve2d
from scipy.ndimage import zoom
import numpy as np


# ======================================================================= #
# GAUSSIAN PSF
# ======================================================================= #
def gaussian_psf(img_ratio):
    """
    Gaussian psf filter function to simulate environment effects from
    surrounding pixels

    Parameters:
        - img_ratio (int): ratio of sensor and ground truth pixel size

    Returns:
        Gaussian PSF (array)
    """

    sigma = 1.5
    convx = np.linspace(-1.5, 1.5, 3)
    convy = np.linspace(-1.5, 1.5, 3)
    convx, convy = np.meshgrid(convx, convy)
    grid = (convx, convy)

    psf = (
        (1 / (2 * np.pi * sigma * sigma))
        * np.exp(-(grid[0]**2 / sigma**2 + grid[1]**2 / sigma**2))
    )

    gaussian_psf = zoom(psf, zoom=img_ratio, order=1)

    return gaussian_psf / gaussian_psf .sum()

# ======================================================================= #
# RECTANGULAR PSF
# ======================================================================= #


def rectangular_psf(img_ratio):
    """
    Rectangular filter to simulate the pixel cell of the sensor

    Parameters:
        - img_ratio (int): ratio of sensor and ground truth pixel size

    Returns:
        - Rectangular PSF (array)
    """

    shape = (img_ratio * 3, img_ratio * 3)
    size = (img_ratio, img_ratio)

    rect_psf = np.zeros(shape, dtype=int)
    start_y = (shape[0] - size[0]) // 2
    start_x = (shape[1] - size[1]) // 2
    rect_psf[start_y:start_y + size[0], start_x:start_x + size[1]] = 1

    return rect_psf.astype(float) / rect_psf.sum()


# ======================================================================= #
# MAIN FUNCTION
# ======================================================================= #
def main(img_ratio):
    """
    Compute both filters and convolve them to produce Net PSF

    Parameters:
        - grid (array): convolution filter window grid
        - gaussian_sigma (float): standard deviation for gaussian
                                    distribution
        - rect_size (int): rectangle sensor pixel size

    Returns:
        - Combine/Net PSF (array): an array that represents sensorn PSF
    """

    # Generate individual PSFs
    gau_psf = gaussian_psf(int(img_ratio))
    rect_psf = rectangular_psf(int(img_ratio))

    # Combine into net PSF via convolution
    net_psf = convolve2d(gau_psf, rect_psf, mode='same')

    return net_psf / net_psf.sum()  # Normalize net PSF


# ======================================================================= #
# INITIALIZE MAIN FUNCTION
# ======================================================================= #
if __name__ == "__main__":
    main()


# ======================================================================= #
# ======================================================================= #
