# ======================================================================= #
"""
hysimu_lrt_computation
------
A function to parallelly compute libRadtran Radiative Transfer Model
using the pyLRT wrapper

RTM code by Mayer et al.
    [http://www.libradtran.org/doku.php?id=start
    C. Emde, R. Buras-Schnell, A. Kylling, B. Mayer, J. Gasteiger,
    U. Hamann, J. Kylling, B. Richter, C. Pause, T. Dowling,
    and L. Bugliaro. The libradtran software package for radiative
    transfer calculations (version 2.0.1). Geoscientific Model
    Development, 9(5):1647-1672, 2016.]
and its python wrapper pyLRT by Gryspeerdt
    [https://github.com/EdGrrr/pyLRT]
"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
import numpy as np
from string import punctuation
from pyLRT import RadTran
from scipy.interpolate import griddata, interp1d
import os
import mat73
from joblib import Parallel, delayed


# ======================================================================= #
# PARALLELIZATION FUNCTION FOR pyLRT
# ======================================================================= #
def lrt_computation(
    index,
    slrt,
    datacube,
    sensor_wavelengths,
    scratch_directory,
    project_name,
    co2_map,
    ch4_map,
    h2o_map,
    dem,
    solar_angles_grid,
    view_angles_grid,
    logger
):
    """
    Output calculation using LRT with
    Loops into datacube and sensor bands, parallelization using joblib

    Parameters:
        - index: datacube pixel indices (i,j,k). 2 spatial indices
        - slrt: LRT class
        - datacube (array, float): surface reflectance datacube
        - sensor_wavelengths (float): array of output/sensor band wavelengths
                                        in nm
        - scratch_directory (str): directory to temporarily save lrt
                                    albedo files
        - project_name (str): project name as identifier
        - co2_map (array): CO2 atmopsheric concentration as input
        - ch4_map (array): CH4 atmopsheric concentration as input
        - h2o_map (array): H2O atmopsheric concentration as input
        - dem (array, float): DEM grid array
        - solar_angles_grid (array, float): a grid of pixel-based solar
                                            angle values
        - view_angles_grid (array, float): a grid of pixel-based view
                                                angle values
        - logger: pass logger

    Returns:
        - output_lrt: LRT computation at-sensor output
    """

    # Indexing through the first 2 dimensions of the datacube
    i, j = index  # Pixel index

    # Reflectance from datacube
    reflectance = np.squeeze(datacube[i, j, :])
    # Make sure reflectance values between 0 and 1
    reflectance = np.clip(reflectance, 0, 1)
    # Mask NaN values from reflectance
    mask = ~np.isnan(reflectance)
    # Interpolate nan values
    int_func = interp1d(
        sensor_wavelengths[mask], reflectance[mask],
        kind='cubic', fill_value='extrapolate'
    )
    reflectance_masked = int_func(sensor_wavelengths)

    # Setup albedo file for LRT input
    alb = reflectance_masked[:, np.newaxis]  # albedo first column
    wave = sensor_wavelengths[:, np.newaxis]  # wavelength second column
    albedo = np.hstack((wave, alb))  # LRT input albedo format

    # Save albedo file for LRT input
    filename = (
        scratch_directory + project_name
        + "_albedo_" + f"{i}" + "_" + f"{j}" + ".dat"
    )
    try:
        np.savetxt(filename, albedo)
    except IOError as e:
        raise RuntimeError(
            f"Failed to save albedo file: {filename}. Error: {e}"
        )
        # errors do not go into logger because joblib

    # Load albedo file for LRT
    slrt.options["albedo_file"] = (filename)

    # Setup gaseous concentration map if set as inputs
    # Changing the mixing ratio of CO2 on the altitude, if
    # gas map is an input
    if co2_map is not None:
        slrt.options["mixing_ratio CO2"] = f"{co2_map[i , j]}"
    else:
        pass

    # Changing the mixing ratio of CH4 on the altitude, if
    # gas map is an input
    if ch4_map is not None:
        slrt.options["mixing_ratio CH4"] = f"{ch4_map[i , j]}"
    else:
        pass

    # Changing the mixing ratio of H2O on the altitude, if
    # gas map is an input
    if h2o_map is not None:
        slrt.options["mol_modify H2O"] = f"{h2o_map[i , j]}"
    else:
        pass

    # Target altitude above MSL in km
    target_altitude = dem[i, j] * 0.001  # based on DEM, which is in m
    if target_altitude < 0:
        logger.warning(
            f"Negative altitude at index {i, j}. Setting to 0.",
            stacklevel=2
        )
        slrt.options["altitude"] = "0"
    else:
        slrt.options["altitude"] = str(target_altitude)

    # Sensor azimuth (the looking direction. 0 deg is sensor looking S)
    view_az = view_angles_grid[i, j, 0]
    slrt.options["phi"] = str(view_az)
    # Cosine of viewing zenith angle
    view_zen = view_angles_grid[i, j, 1]
    slrt.options["umu"] = str(np.cos(np.deg2rad(view_zen)))

    # Solar angles
    # Solar azimuth
    sun_az = solar_angles_grid[i, j, 0]
    sun_az %= 360  # normalize azimuth
    if 0 <= sun_az < 180:
        phi0 = sun_az + 180
    elif 180 <= sun_az <= 360:
        phi0 = sun_az - 180
    slrt.options["phi0"] = str(phi0)  # sun azimuth (180 deg is sun in N)

    # Solar zenith
    sun_zen = solar_angles_grid[i, j, 1]
    if sun_zen >= 90:
        sun_zen = 89.9999
        # if the sun is behind the plane, set angel #to 89.9999 to prevent
        # error
    slrt.options["sza"] = str(sun_zen)  # sun zenith angle

    try:
        sdata, sverb = slrt.run(verbose=True)

        output_lrt = griddata(
            sdata[:, 0], sdata[:, 1], sensor_wavelengths, method="nearest"
        )  # sample LRT output to sensor bands

        os.remove(filename)
    except Exception as e:
        raise RuntimeError(
            f"Error occurred during LRT parallel computation: {e}"
        )
        # error does not go into logger because joblib

    return output_lrt


# ======================================================================= #
# MAIN FUNCTION FOR pyLRT
# ======================================================================= #
def main(
    lrt_path,
    output_type,
    scratch_directory,
    project_name,
    datacube,
    aero,
    atmos,
    h2o_mm,
    co2_ppm,
    ch4_ppm,
    sensor_wavelengths,
    platform_type,
    sensor_altitude,
    datetime,
    cores,
    dem,
    solar_angles_grid,
    view_angles_grid,
    logger,
    verbose_level
):
    """
    Main function to setup LRT arguments and inputs and then loop over
    the datacube to run LRT pixel by pixel

    Parameters:
        - lrt_path (str): libradtran installation path
        - output_type (str): lrt output type
        - scratch_directory (str): directory to temporarily save lrt
                                    albedo files
        - project_name (str): project name as identifier
        - datacube (array, float): surface reflectance datacube
        - aero (int): libradtran aerosol parameter
        - atmos (str): libradtran atmospheric profile parameter
        - h2o_mm (float/str): CO2 atmopsheric concentration as input
        - co2_ppm (float/str): CH4 atmopsheric concentration as input
        - ch4_ppm (float/str): H2O atmopsheric concentration as input
        - sensor_wavelengths (float): array of output/sensor band wavelengths
                                        in nm
        - platform_type (str): platform type (satellite, uav, aircraft)
        - sensor_altitude (float): sensor/platform altitude in km
        - datetime (str): datetime string
        - cores (int): how many processing cores to run joblib on
        - dem (array, float): DEM grid array
        - solar_angles_grid (array, float): a grid of pixel-based solar
                                            angle values
        - view_angles_grid (array, float): a grid of pixel-based view
                                                angle values
        - logger: pass logger
        - verbose_level (int): joblib verbose level to track proggress

    Returns:
        - output_datacube (array, float): a datacube array of LRT output.
                                            at-sensor datacube with
                                            atmospheric effects
    """

    # Instantiate libRadtran
    slrt = RadTran(lrt_path)

    # Setup RTE solver
    slrt.options["rte_solver"] = "disort"
    slrt.options["source"] = "solar"
    slrt.options["mol_abs_param"] = "reptran"
    slrt.options["wavelength"] = (
        f"{sensor_wavelengths[0] + 10}"
        " "
        f"{sensor_wavelengths[-1] - 10}"
    )

    # Setup aerosol parameter
    # Check if aero parameter is valid
    if isinstance(aero, int) and aero in [1, 4, 5, 6]:  # LRT aero haze list
        slrt.options["aerosol_default"] = ""
        slrt.options["aerosol_haze"] = str(aero)
    # Otherwise, set default aerosol
    elif aero == "default":
        slrt.options["aerosol_default"] = ""
    else:
        slrt.options["aerosol_default"] = ""
        logger.warning(
            "Invalid LRT low aerosol type. Default low aerosol is applied.",
            stacklevel=2
        )

    # Setup atmospheric profile parameters
    lrt_atmos_list = [
        "tropics", "midlatitude_summer",
        "midlatitude_winter", "subarctic_summer",
        "subarctic_winter", "US-standard"
    ]  # LRT atmospheric profile list
    # Check if atmos parameter is valid
    if atmos in lrt_atmos_list:
        slrt.options["atmosphere_file"] = str(atmos)
    # Otherwise, set default atmosphere
    elif atmos == "default":
        slrt.options["atmosphere_file"] = "US-standard"
    else:
        logger.warning(
            "Invalid LRT atmospheric profile. US-standard is applied.",
            stacklevel=2
        )
        slrt.options["atmosphere_file"] = "US-standard"

    # Setup gases parameters

    # if CO2 concentration is a constant at target altitude
    # and will be scaled for the entire atmospheric column
    if isinstance(co2_ppm, (int, float)):
        slrt.options["mixing_ratio CO2"] = str(co2_ppm)
    # Else if CO2 concentration is a distributed in a grid map
    elif (
        isinstance(co2_ppm, str)
        and os.path.isfile(co2_ppm)
        and co2_ppm != "default"
    ):
        # CO2 concentration as a pixel based map file
        if co2_ppm.endswith(".mat"):
            data_dict = mat73.loadmat(co2_ppm)
            co2_map = np.array(list(data_dict.values()))
            co2_map = np.squeeze(co2_map)
        elif co2_ppm.endswith(".npy"):
            co2_map = np.load(co2_ppm)
        else:
            raise ValueError(
                "Invalid input CO2 file type")
    # Else if default
    elif co2_ppm == "default":
        pass
    # If invalid, set as default
    else:
        raise ValueError(
            "Gas input should be a float/int/path to file/'default'"
        )
    # Check gas concentration map to pass to child function
    try:
        co2_map
    except NameError:
        co2_map = None

    # if CH4 concentration is a constant at target altitude
    # and will be scaled for the entire atmospheric column
    if isinstance(ch4_ppm, (int, float)):
        slrt.options["mixing_ratio CH4"] = str(ch4_ppm)
    # Else if CH4 concentration is a distributed in a grid map
    elif (
        isinstance(ch4_ppm, str)
        and os.path.isfile(ch4_ppm)
        and ch4_ppm != "default"
    ):
        # CH4 concentration as a pixel based map file
        if ch4_ppm.endswith(".mat"):
            data_dict = mat73.loadmat(ch4_ppm)
            ch4_map = np.array(list(data_dict.values()))
            ch4_map = np.squeeze(ch4_map)
        elif ch4_ppm.endswith(".npy"):
            ch4_map = np.load(ch4_ppm)
        else:
            raise ValueError(
                "Invalid input CH4 file type")
    # Else if default
    elif ch4_ppm == "default":
        pass
    # If invalid, set as default
    else:
        raise ValueError(
            "Gas input should be a float/int/path to file/'default'"
        )
    # Check gas concentration map to pass to child function
    try:
        ch4_map
    except NameError:
        ch4_map = None

    # if H2O concentration is a constant at target altitude
    # and will be scaled for the entire atmospheric column
    if isinstance(h2o_mm, (int, float)):
        slrt.options["mol_modify H2O"] = str(h2o_mm) + " MM"
    # Else if H2O concentration is a distributed in a grid map
    elif (
        isinstance(h2o_mm, str)
        and os.path.isfile(h2o_mm)
        and h2o_mm != "default"
    ):
        # H2O concentration as a pixel based map file
        if h2o_mm.endswith(".mat"):
            data_dict = mat73.loadmat(h2o_mm)
            h2o_map = np.array(list(data_dict.values()))
            h2o_map = np.squeeze(h2o_map)
        elif h2o_mm.endswith(".npy"):
            h2o_map = np.load(h2o_mm)
        else:
            raise ValueError(
                "Invalid input H2O file type")
    # Else if default
    elif h2o_mm == "default":
        pass
    # If invalid, set as default
    else:
        raise ValueError(
            "Gas input should be a float/int/path to file/'default'"
        )
    # Check gas concentration map to pass to child function
    try:
        h2o_map
    except NameError:
        h2o_map = None

    # Setup geometrical parameters
    if platform_type == "satellite":
        slrt.options["zout"] = "TOA"  # top of atmosphere
    else:
        sensor_altitude *= 0.001
        slrt.options["zout"] = str(sensor_altitude)

    # Change datetime string to match lrt format
    lrt_time = datetime
    for char in punctuation:
        lrt_time = lrt_time.replace(char, " ")
    slrt.options["time"] = lrt_time

    # Setup output and post-processing
    slrt.options["output_user"] = "lambda uu"  # band & radiance value
    # If reflectance value is desired
    if output_type == "reflectance":
        slrt.options["output_quantity"] = "reflectivity"

    # Parallel computation of LRT
    # Get element spatial indices
    indices = [
        (i, j)
        for i in range(datacube.shape[0])
        for j in range(datacube.shape[1])
    ]

    # Parallel computation to produce the output
    try:
        output_values = Parallel(n_jobs=cores, verbose=verbose_level)(
            delayed(lrt_computation)(
                index=index,
                slrt=slrt,
                datacube=datacube,
                sensor_wavelengths=sensor_wavelengths,
                scratch_directory=scratch_directory,
                project_name=project_name,
                co2_map=co2_map,
                ch4_map=ch4_map,
                h2o_map=h2o_map,
                dem=dem,
                solar_angles_grid=solar_angles_grid,
                view_angles_grid=view_angles_grid,
                logger=logger
            )
            for index in indices
        )
    except Exception as e:
        logger.error(
            f"Error occurred during RTM computation. Error: {e}"
        )

    # Reshape the radiance values back to the shape of reflectance array
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
