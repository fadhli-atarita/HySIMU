# ======================================================================= #
"""
hysimu_sixs_computation
------
A function to parallelly compute 6S Radiative Transfer Model
using the py6S wrapper.

6S (Second Simulation of a Satellite Signal in the Solar) installation.
    Spectrum vector code by Vermote et al.
        [https://salsa.umd.edu/6spage.html
        Vermote, E.F., Tanré, D., Deuzé, J.L., Herman, M.,
        & Morcrette, J.-J. (1997), Second Simulation of the Satellite
        Signal in the Solar Spectrum, 6S: An Overview,
        IEEE Transactions on Geoscience and Remote Sensing,
        Vol. 35, No. 3, p. 675-686.]

and its python wrapper py6S by Wilson
        [https://py6s.readthedocs.io/en/latest/
        Wilson, R. T. (2013). Py6S: A Python interface to the 6S
        radiative transfer model. Computers & Geosciences, 51, 166–171.
        https://doi.org/10.1016/j.cageo.2012.08.002]
"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
import numpy as np
from Py6S import *
from joblib import Parallel, delayed
import dateutil.parser


# ======================================================================= #
# SIXS COMPUTATION FUNCTION
# ======================================================================= #
def sixs_computation(
    sixs,
    reflectance,
    wave,
    target_alt,
    solar_az,
    solar_zen,
    view_az,
    view_zen,
    output_type,
):
    """
    Compute 6S using given parameters

    Parameters:
        - sixs: 6S class
        - reflectance (float): reflectance value [0,1] at the specific
                                pixel in the datacube
        - wave (float): wavelength in nm
        - target_alt (float): altitude of the pixel in km
        - solar_az (float): solar azimuth angle
        - solar_zen (float): solar zenith angle
        - view_az (float): sensor view azimuth angle
        - view_zen (float): sensor view zenith angle
        - output_type (str): 6S output type

    Returns:
        - sixs.outputs.values: 6S computation at-sensor output
    """

    # Set 6S parameters
    # Target altitude
    sixs.altitudes.set_target_custom_altitude(target_alt)
    # Check solar azimuth value
    sixs.geometry.solar_a = solar_az
    if solar_zen >= 90:  # prevent error for sun behind the plane
        # This takes care of terrain shadowing because the radiance
        # computation will be almost 0
        solar_zen = 89.9999
    # View geometry
    sixs.geometry.view_a = view_az
    sixs.geometry.view_z = view_zen

    # Make sure reflectance value is between 0 and 1
    reflectance = max(min(1, reflectance), 0)
    sixs.ground_reflectance = GroundReflectance.HomogeneousLambertian(
        reflectance)
    sixs.wavelength = Wavelength(wave)

    # Compute 6S outputs and catch any error
    try:
        sixs.run()  # Run 6S
        return sixs.outputs.values[output_type]
    except Exception as e:
        raise RuntimeError(
            f"Error occurred during 6S parallel computation: {e}"
        )
        # Errors cannot go into logger because joblib


# ======================================================================= #
# SIXS INDEXING FUNCTION
# ======================================================================= #
def sixs_output(
    index,
    sixs,
    datacube,
    sensor_wavelengths,
    solar_angles_grid,
    view_angles_grid,
    dem,
    output_type,
):
    """
    Indexing function, to loop through datacube and bands as inputs for 6S

    Parameters:
        - index: datacube pixel indices (i,j,k). 2 spatial and 1 spectral
                    indices
        - sixs: 6S class
        - datacube (array, float): surface reflectance datacube
        - sensor_wavelengths (float): array of output/sensor band wavelengths
                                        in nm
        - solar_angles_grid (array, float): a grid of pixel-based solar
                                                angle values
        - view_angles_grid (array, float): a grid of pixel-based view
                                                angle values
        - dem (array, float): DEM grid array
        - output_type (str): 6S output type

    Returns:
        - sixs.outputs.values: index-based parallellization of
                                6S computation at-sensor output values
    """

    i, j, k = index  # indexing

    # Run the computation function looping through the indices
    try:
        return sixs_computation(
            sixs=sixs,
            reflectance=datacube[i, j, k],
            wave=sensor_wavelengths[k] * 0.001,
            target_alt=dem[i, j] * 0.001,
            solar_az=solar_angles_grid[i, j, 0],  # azimuth is at 0
            solar_zen=solar_angles_grid[i, j, 1],  # zenith is at 0
            view_az=view_angles_grid[i, j, 0],  # azimuth is at 0
            view_zen=view_angles_grid[i, j, 1],  # zenith is at 1
            output_type=output_type,
        )
    except Exception as e:
        raise RuntimeError(
            f"Error occurred during 6S indexing. Error: {e}"
        )
        # Errors cannot go into logger because joblib


# ======================================================================= #
# MAIN FUNCTION for py6S
# ======================================================================= #
def main(
    sixS_path,
    project_name,
    aero,
    atmos,
    latitude,
    date,
    output_type,
    platform_type,
    sensor_altitude,
    datacube,
    sensor_wavelengths,
    solar_angles_grid,
    view_angles_grid,
    dem,
    cores,
    logger,
    verbose_level
):
    """
    Main function, setup 6S inputs and parallelly loop through datacube
    using joblib

    Parameters:
        - sixS_path (str): 6S installation path
        - project_name (str): project name as identifier
        - aero (str): 6S aerosol parameter
        - atmos (str): 6S atmospheric profile parameter
        - latitude (float): latitude of reference point, needed if atmos
                            is auto
        - date (str): date of measurement, needed if atmos is auto
        - output_type (str): 6S output type
        - platform_type (str): platform type (satellite, uav, aircraft)
        - sensor_altitude (float): sensor/platform altitude in km
        - datacube (array, float): surface reflectance datacube
        - sensor_wavelengths (float): array of output/sensor band wavelengths
                                        in nm
        - solar_angles_grid (array, float): a grid of pixel-based solar
                                                angle values
        - view_angles_grid (array, float): a grid of pixel-based view
                                                angle values
        - dem (array, float): DEM grid array
        - cores (int): how many processing cores to run joblib on
        - logger: pass logger
        - verbose_level (int): joblib verbose level to track proggress

    Returns:
        - output_datacube (array, float): a datacube array of 6S output.
                                            at-sensor datacube with
                                            atmospheric effects
    """

    # Instantiate 6S
    if sixS_path is None or sixS_path.lower() == "default":
        sixs = SixS()
    else:
        sixs = SixS(sixS_path)

    # Setup atmospheric parameters
    # Get default 6S aerosol profile list
    sixS_aero_list = [
        attr for attr in dir(AeroProfile())
        if not callable(getattr(AeroProfile(), attr))
        and not attr.startswith("__")
    ]
    # Check input aerosol. if valid, used as input
    if aero in sixS_aero_list:
        sixs.aero_profile = AeroProfile.PredefinedType(
            getattr(AeroProfile, aero)
        )
    # Otherwise, set up default aerosol
    else:
        logger.warning(
            "Invalid 6S aerosol profile. "
            "Defaulting to NoAerosols.",
            stacklevel=2
        )
        sixs.aero_profile = AeroProfile.PredefinedType(
            AeroProfile.NoAerosols)

    # Get default 6S atmospheric profile list
    sixS_atmos_list = [
        attr for attr in dir(AtmosProfile())
        if not callable(getattr(AtmosProfile(), attr))
        and not attr.startswith("__")
    ]
    # Check input atmospheric profile. if valid, used as input
    if atmos in sixS_atmos_list:
        sixs.atmos_profile = AtmosProfile.PredefinedType(
            getattr(AtmosProfile, atmos)
        )
    # if auto, setup latitude and date parameters
    elif atmos.lower() == "auto":
        dt = dateutil.parser.parse(date)
        date_ymd = f"{dt.day}/{dt.month}/{dt.year}"
        sixs.atmos_profile = AtmosProfile.FromLatitudeAndDate(
            latitude, date_ymd
        )
    # Otherwise, set up default atmosphere
    else:
        logger.warning(
            "Invalid 6S atmospheric profile. "
            "Defaulting to NoGaseousAbsorption.",
            stacklevel=2
        )
        sixs.atmos_profile = AtmosProfile.PredefinedType(
            AtmosProfile.NoGaseousAbsorption
        )

    # Sensor altitudes
    if platform_type.lower() == "satellite":
        sixs.altitudes.set_sensor_satellite_level()
    else:
        sensor_altitude *= 0.001
        sixs.altitudes.set_sensor_custom_altitude(sensor_altitude)

    # Start 6S RTM calculation
    # Indices for all elements in the reflectance cube
    indices = [
        (i, j, k)
        for i in range(datacube.shape[0])
        for j in range(datacube.shape[1])
        for k in range(datacube.shape[2])
    ]

    # Parallel computation to produce the output
    try:
        logger.info("Starting 6S indexing.")

        output_values = Parallel(n_jobs=cores, verbose=verbose_level)(
            delayed(sixs_output)(
                index=index,
                sixs=sixs,
                datacube=datacube,
                sensor_wavelengths=sensor_wavelengths,
                solar_angles_grid=solar_angles_grid,
                view_angles_grid=view_angles_grid,
                dem=dem,
                output_type=output_type,
            )
            for index in indices
        )
    except Exception as e:
        logger.error(
            f"Error occurred during RTM computation. Error: {e}"
        )

    # Reshape the output values back to the shape of the reflectance
    # datacube array
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
