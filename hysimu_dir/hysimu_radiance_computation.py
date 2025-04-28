# ======================================================================= #
"""
hysimu_radiance_computation
------
A function to parallelly compute radiance without radiative transfer
model based on the formula:
    Reflectance(λ) = π * TOA Radiance(λ) /
                 Solar irradiance(λ) * cos(sun zenith)
    [Chander, G., Markham, B. L., & Helder, D. L. (2009).
    Summary of current radiometric calibration coefficients for Landsat
    MSS, TM, ETM+, and EO-1 ALI sensors. Remote Sensing of Environment,
    113(5), 893–903. https://doi.org/10.1016/j.rse.2009.01.007]

Solar irradiance values are taken from the "Global" dataset of
the ASTM G173-03 Standard Spectrum included in the pvlib library.
    [G03 Committee. (n.d.).
    Tables for Reference Solar Spectral Irradiances:
    Direct Normal and Hemispherical on 37 Tilted Surface. ASTM International.
    https://doi.org/10.1520/G0173-03R08]

"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
import numpy as np
from scipy.interpolate import griddata
import pvlib
from joblib import Parallel, delayed


# ======================================================================= #
# RADIANCE COMPUTATION FUNCTION
# ======================================================================= #
def radiance_computation(
    reflectance,
    irradiance,
    solar_zen,
):
    """
    Calculate radiance using given parameters

    Parameters:
        - reflectance (float): reflectance value [0,1] at the specific
                                pixel in the datacube
        - irradiance (float): solar irradiance at the specific wavelength
        - solar_zen (float): solar zenith angle

    Returns:
        - rad: radiance computation value (float)
    """
    try:
        rad = (
            1000 * reflectance * np.cos(np.deg2rad(solar_zen))
            * irradiance / np.pi
        )
        return rad
    except Exception as e:
        raise RuntimeError(
            f"Error occurred during parallel radiance computation: {e}"
        )
        # error does not go into logger because joblib


# ======================================================================= #
# RADIANCE INDEXING FUNCTION
# ======================================================================= #
def radiance_output(
    index,
    datacube,
    irradiance_all_bands,
    solar_angles_grid,
):
    """
    Indexing function, to loop through datacube and bands

    Parameters:
        - index (list, int): list of indices to loop through the
                                datacube
        - datacube (array, float): surface reflectance datacube
        - irradiance_all_bands (float): solar irradiance for all sensor
                                        bands
        - solar_angles_grid (array, float): a grid of pixel-based solar
                                                angle values

    Returns:
        - radiance computation: indexing calculation output
    """
    i, j, k = index  # indexing

    try:
        return radiance_computation(
            reflectance=datacube[i, j, k],
            irradiance=irradiance_all_bands[k],
            solar_zen=solar_angles_grid[i, j, 1],
        )
    except Exception as e:
        raise RuntimeError(
            f"Error occurred during radiance indexing. Error: {e}"
        )
        # error does not go into logger because joblib


# ======================================================================= #
# MAIN FUNCTION
# ======================================================================= #
def main(
    datacube,
    solar_angles_grid,
    sensor_wavelengths,
    cores,
    logger,
    verbose_level,
):
    """
    Main function, to simply calculate radiance values from reflectance
    datacube using default parameters

    Parameters:
        - datacube (array, float): surface reflectance datacube
        - solar_angles_grid (array, float): a grid of pixel-based solar
                                                angle values
        - sensor_wavelengths (float): array of output/sensor band wavelengths
                                        in nm
        - cores (int): how many processing cores to run joblib on
        - logger: pass logger
        - verbose_level (int): joblib verbose level to track proggress

    Returns:
        - output_datacube (array, float): a datacube array of 6S output.
                                            at-sensor datacube with
                                            atmospheric effects
     """

    # Get solar irradiance values for all wavelengths from pvlib
    am15 = pvlib.spectrum.get_reference_spectra(standard="ASTM G173-03")

    # Exoatmosphere or extraterrestrial solar spectral irradiance
    ext_irr = np.array(am15["global"])
    wave = np.array(am15.index)

    # Resampled to sensor bands
    ext_irr_resampled = griddata(
        wave, ext_irr, sensor_wavelengths, method="nearest"
    )

    # Joblib loop indices through the reflectance datacube
    indices = [
        (i, j, k)
        for i in range(datacube.shape[0])
        for j in range(datacube.shape[1])
        for k in range(datacube.shape[2])
    ]

    # Parallel computation to produce the output
    try:
        logger.info("Starting radiance indexing.")

        output_values = Parallel(n_jobs=cores, verbose=verbose_level)(
            delayed(radiance_output)(
                index=index,
                datacube=datacube,
                irradiance_all_bands=ext_irr_resampled,
                solar_angles_grid=solar_angles_grid,
            )
            for index in indices
        )
    except Exception as e:
        logger.error(
            f"Error occurred during RTM computation. Error: {e}"
        )

    # Reshape the radiance values back to the shape of reflectance datacube
    output_datacube = np.array(
        output_values).reshape(datacube.shape)

    return output_datacube


# ======================================================================= #
# INITIALIZE MAIN FUNCTION
# ======================================================================= #
if __name__ == "__main__":
    main()


# ======================================================================= #
# ======================================================================= #
