# ======================================================================= #
"""
hysimu_psf_simplified
A function to compute a simplified PSF for hyperspectral imaging.
A simplified PSF filter based on a gaussian distribution to simulate
an airy disk convolved with a rectangular response function
image motion and electronic response are ignored.

Calculation of the PSFs follows formulas provided by:
Schowengerdt, R. A. (2007). Remote sensing, models, and
    methods for image processing (3rd ed). Academic Press. p85-91
"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
from scipy.signal import convolve2d
import numpy as np


# ======================================================================= #
# GAUSSIAN PSF
# ======================================================================= #
def gaussian_psf(grid, sigma):
    """
    Gaussian psf filter function to simulate environment effects from
    surrounding pixels

    Parameters:
        - sigma (float): standard deviation for gaussian distribution
        - grid (array): convolution filter window grid

    Returns:
        Gaussian PSF (array)
    """

    psf = np.exp(-(grid[0]**2 + grid[1]**2) / (2 * sigma**2))
    return psf / psf.sum()

# ======================================================================= #
# RECTANGULAR PSF
# ======================================================================= #


def rectangular_psf(grid, rect_size):
    """
    Rectangular filter to simulate the pixel cell of the sensor

    Parameters:
        - grid (array): convolution filter window grid
        - rect_size (int): rectangle sensor pixel size

    Returns:
        - Rectangular PSF (array)
    """

    psf = (
        (np.abs(grid[0]) <= rect_size / 2)
        & (np.abs(grid[1]) <= rect_size / 2)
    )
    return psf.astype(float) / psf.sum()


# ======================================================================= #
# MAIN FUNCTION
# ======================================================================= #
def main(grid, gaussian_sigma, rect_size):
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
    gau_psf = gaussian_psf(grid, gaussian_sigma)
    rect_psf = rectangular_psf(grid, rect_size)

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
