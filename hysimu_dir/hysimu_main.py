# ======================================================================= #
# ======================================================================= #
"""
HySIMUv2.0
------
A hyperspectral remote sensing forward modelling toolkit
    Author: Fadhli Atarita
    Year: 2025
    Version: 2.0

Create a synthetic at-sensor hyperspectral reflectance/radiance datacube
by simulating ground truth datasets based on:
    - Sensor parameters
    - Mission parameters
    - Surface parameters
    - Atmospheric parameters (for Radiative Transfer Model)

HySIMU consists of the main "hysimu_main.py" script and several
functions, all located inside the hysimu directory ("/hysimu_dir")
    - "hysimu_logging.py"
    - "hysimu_random_map_generator.py"
    - "hysimu_random_spectra_selector.py"
    - "hysimu_spectral_texture.py"
    - "hysimu_solar_geometry_computation.py"
    - "hysimu_sixs_computation.py"
    - "hysimu_lrt_computation.py"
    - "hysimu_radiance_computation.py"
    - "hysimu_psf_filter_simplified.py"
    - "hysimu_spatial_mixing.py"

For the full documentation, read these files (inside the "/hysimu_files"
directory:
    - README[HySIMU]
    - INPUTGUIDE[HySIMU]
    - DEVNOTES[HySIMU]
"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
import numpy as np
import mat73
import pandas as pd
import sys
import spectral as spy
import gzip
import hysimu_logging as hy_log
import time
start_time = time.time()


# ======================================================================= #
# READ PROJECT INFO
# ======================================================================= #
# Read input .xlsx file
input_file = sys.argv[1]
df = pd.read_excel(input_file)
# df = pd.read_csv(input_file, sep=';')

# Directories
project_dir = df.loc[
    df["parameter"] == "project_directory_path",
    "value"].iloc[0]

# Project and Mission info
project_name = df.loc[
    df["parameter"] == "project_name",
    "value"].iloc[0]
platform_type = df.loc[
    df["parameter"] == "platform_type",
    "value"].iloc[0]
sensor_name = df.loc[
    df["parameter"] == "platform_or_sensor_identifier",
    "value"].iloc[0]

# Processing cores for joblib parallelization
cores = df.loc[
    df["parameter"] == "processing_cores",
    "value"].iloc[0]
cores = int(cores)

# Setup joblib verbose because logging is not propagated into joblib
# Catch errors or track progress using verbose instead
verbose_lvl = df.loc[
    df["parameter"] == "joblib_verbose_level",
    "value"].iloc[0]
verbose_lvl = int(verbose_lvl)


# ======================================================================= #
# SETUP LOGGING SYSTEM
# ======================================================================= #
logger = hy_log.main(project_dir, project_name)
logger.info("HySIMU - START")


# ======================================================================= #
# CREATE GROUND TRUTH / SPECTRAL DISTRIBUTION (ZONES) MAP
# ======================================================================= #
logger.info("Creating ground truth map.")

# Spatial resolutions
# Pixel resolution of the ground truth spectral map
gt_res = df.loc[
    df["parameter"] == "spectral_map_pixel_resolution_in_m",
    "value"].iloc[0]
# Sensor spatial resolution
sensor_res = df.loc[
    df["parameter"] == "sensor_spatial_resolution_in_m",
    "value"].iloc[0]

# Ground truth spectral map file (input or random)
spectral_map_file = df.loc[
    df["parameter"] == "spectral_map_file",
    "value"].iloc[0]
# Complexity level of the randomly generated spectral distribution map
complex_lvl = df.loc[
    df["parameter"] == "spectral_map_complexity_level",
    "value"].iloc[0]
# Number of endmembers in the scene (input or randomly selected)
num_endmembers = df.loc[
    df["parameter"] == "number_of_endmembers",
    "value"].iloc[0]
# Number of subregions within each spectral zone that represent
# spectral texture
num_subregions = df.loc[
    df["parameter"] == "number_of_subregions",
    "value"].iloc[0]
# Random factor. Needed if spectral zone map is generated from an input DEM
random_factor = df.loc[
    df["parameter"] == "spectral_map_randomness_offset_factor",
    "value"].iloc[0]

# Input DEM file, if any
input_dem_file = df.loc[
    df["parameter"] == "input_DEM_file",
    "value"].iloc[0]

# DEM options
if input_dem_file.endswith(".mat"):
    # If DEM file is a matlab v7.3 file
    data_dict = mat73.loadmat(input_dem_file)
    dem_i = np.array(list(data_dict.values()))
    dem_i = np.squeeze(dem_i)
    DEM_option = "input"
elif input_dem_file.endswith(".npy"):
    # If DEM file is a numpy file
    dem_i = np.load(input_dem_file)
    DEM_option = "input"
# If spectral map type is random
elif input_dem_file.lower() == "random":
    DEM_option = "random"
else:
    logger.warning(
        "None or invalid DEM. "
        "Flat DEM is assumed."
    )
    DEM_option = "none"

# Spectral distribution/zone map
# If spectral zone map is an input file
if spectral_map_file.endswith(".mat"):
    # If it is a matlab file v7.3
    data_dict = mat73.loadmat(spectral_map_file)
    spectral_map = np.array(list(data_dict.values()))
    spectral_map = np.squeeze(spectral_map)
    num_row = spectral_map.shape[0]
    num_col = spectral_map.shape[1]
elif spectral_map_file.endswith(".npy"):
    # If it is a numpy file
    spectral_map = np.load(spectral_map_file)
    num_row = spectral_map.shape[0]
    num_col = spectral_map.shape[1]
elif spectral_map_file.lower() == "dem_based_fractal":

    # If spectral map type is DEM-based
    # Spectral zone map will be generated based on input DEM
    # Requires DEM to be an input to the dem_i variable
    DEM_option = "dem_based_fractal"

    # Import map generator module
    import hysimu_random_map_generator as hy_map

    # Get DEM dimensions
    num_row = dem_i.shape[0]
    num_col = dem_i.shape[1]

    # Set the noise/smooth levels of each spectral region
    spatial_smoothness = df.loc[
        df["parameter"] == "spatial_smoothness_list",
        "value"].iloc[0]
    # Check the values from the input list
    try:
        spatial_smoothness = spatial_smoothness.split(',')
        spatial_smoothness = [float(var) for var in spatial_smoothness]
    except Exception:
        spatial_smoothness = np.ones(num_endmembers)
        logger.warning(
            "Invalid spatial smoothness values. "
            "1 is assumed for all spectral zones."
        )

    # Run hysimu_random_map_generator main function
    # max_height is None because DEM is already prescribed
    # dem_r will be ignored
    spectral_map, dem_r, subregion_map = hy_map.main(
        num_row=num_row,
        num_col=num_col,
        complex_level=complex_lvl,
        num_endmembers=num_endmembers,
        num_subregions=num_subregions,
        max_height=None,
        DEM_option=DEM_option,
        DEM_input=dem_i,
        random_factor=random_factor,
        spatial_smoothness=spatial_smoothness
    )

    # Save spectral distribution map as a gzip file
    f = gzip.GzipFile(
        project_dir + "/" + project_name + "_spectral_map.npy.gz",
        "w")
    np.save(file=f, arr=spectral_map)
    f.close()

    # Reassign DEM variable
    dem_map = dem_i

elif spectral_map_file.lower() == "random":

    # If spectral map type is random

    # Spectral map will be generated randomly from a fractal field
    # generated from inverse FFT of Power Spectral Density of
    # FFT frequencies
    import hysimu_random_map_generator as hy_map

    # Get dimensions as inputs
    num_row = df.loc[
        df["parameter"] == "number_of_rows",
        "value"].iloc[0]
    num_col = df.loc[
        df["parameter"] == "number_of_columns",
        "value"].iloc[0]

    # If DEM will be generated from the same random field,
    # maximum altitude for the DEM is an input
    max_alt = df.loc[
        df["parameter"] == "DEM_maximum_altitude",
        "value"].iloc[0]

    # Set the noise/smooth levels of each spectral region
    spatial_smoothness = df.loc[
        df["parameter"] == "spatial_smoothness_list",
        "value"].iloc[0]
    # Check the values from the input list
    try:
        spatial_smoothness = spatial_smoothness.split(',')
        spatial_smoothness = [float(var) for var in spatial_smoothness]
    except Exception:
        spatial_smoothness = np.ones(num_endmembers)
        logger.warning(
            "Invalid spatial smoothness values. "
            "1 is assumed for all spectral zones."
        )

    # Run hysimu_random_map_generator main function
    # DEM is None because DEM is either randomly generated
    # or already prescribed as input
    spectral_map, dem_r, subregion_map = hy_map.main(
        num_row=num_row,
        num_col=num_col,
        complex_level=complex_lvl,
        num_endmembers=num_endmembers,
        num_subregions=num_subregions,
        max_height=max_alt,
        DEM_option=DEM_option,
        DEM_input=None,
        random_factor=random_factor,
        spatial_smoothness=spatial_smoothness
    )

    # Save spectral distribution map as a gzip file
    f = gzip.GzipFile(
        project_dir + "/" + project_name + "_spectral_map.npy.gz",
        "w")
    np.save(file=f, arr=spectral_map)
    f.close()

    # Check if DEM already prescribed as input or generated
    # dem_i: input | dem_r: randomly generated
    try:
        # If DEM exists as an input, assign dem_i as dem_map
        dem_map = dem_i if dem_i else dem_r
    except NameError:
        # Catch the scenario if dem_i doesn't exist. Assign
        # dem_r as dem_map
        dem_map = dem_r if "dem_r" in globals() else None

    # Save the generated DEM
    f = gzip.GzipFile(
        project_dir + "/" + project_name + "_dem_map.npy.gz",
        "w")
    np.save(file=f, arr=dem_map)
    f.close()

else:
    raise ValueError(
        "Spectral map parameter invalid"
    )


# ======================================================================= #
# READ/SELECT SPECTRAL ENDMEMBERS
# ======================================================================= #
logger.info("Reading/selecting spectral endmembers.")

# Spectral endmembers file (input or random)
spectral_endmembers_file = df.loc[
    df["parameter"] == "spectral_endmembers_file",
    "value"].iloc[0]

# Bands & Sensor band wavelengths
# Make sure both band files are in nanometers

# Input/original bands
og_bands = df.loc[
    df["parameter"] == "original_bands_file",
    "value"].iloc[0]
# If the spectral are randomly selected using the random_spectra_selector
# module, original bands file is not needed
if og_bands.lower() == 'none':
    pass
else:
    og_bands = np.loadtxt(og_bands)

# Original/input bands FHWM file
og_bands_fwhm = str(df.loc[
    df["parameter"] == "original_bands_fwhm_file",
    "value"].iloc[0])
if og_bands_fwhm.endswith(".txt") or og_bands_fwhm.endswith(".dat"):
    og_bands_fwhm = np.loadtxt(og_bands_fwhm)
# Else if FWHM not specified
else:
    logger.info(
        "No FWHM file specified for original bands. "
        "FWHM is assumed to be half distance."
    )
    og_bands_fwhm = None

# Output/sensor bands file
sensor_bands = np.loadtxt(
    df.loc[
        df["parameter"] == "sensor_bands_file",
        "value"].iloc[0])

# Output/sensor bands FWHM file
sensor_bands_fwhm = str(df.loc[
    df["parameter"] == "sensor_bands_fwhm_file",
    "value"].iloc[0])
if sensor_bands_fwhm.endswith(".txt") or sensor_bands_fwhm.endswith(".dat"):
    sensor_bands_fwhm = np.loadtxt(sensor_bands_fwhm)
# Else if FWHM not specified
else:
    logger.info(
        "No FWHM file specified for sensor bands. "
        "FWHM is assumed to be half distance."
    )
    sensor_bands_fwhm = None

# Endmember spectral signatures input file
if spectral_endmembers_file.endswith(".mat"):
    # If endmember signatures contained in a matlab v7.3 file
    data_dict = mat73.loadmat(spectral_endmembers_file)
    og_endmembers = np.array(list(data_dict.values()))
    og_endmembers = np.squeeze(og_endmembers)

    # Resample endmembers from original to sensor bands
    resampler = spy.BandResampler(
        centers1=og_bands,
        centers2=sensor_bands,
        fwhm1=og_bands_fwhm,
        fwhm2=sensor_bands_fwhm
    )
    endmembers = resampler(og_endmembers)

elif spectral_endmembers_file.endswith(".npy"):
    # If endmember signatures contained in a matlab v7.3 file
    sensor_bands = sensor_bands

    og_endmembers = np.load(spectral_endmembers_file)

    resampler = spy.BandResampler(
        centers1=og_bands,
        centers2=sensor_bands,
        fwhm1=og_bands_fwhm,
        fwhm2=sensor_bands_fwhm
    )
    endmembers = resampler(og_endmembers)

# Else if endmember spectral signatures are randomly generated
elif spectral_endmembers_file.lower() == "random":
    logger.info(
        "Randomly selecting spectral endmembers from "
        "the ECOSTRESS library."
    )

    # If endmember spectra is set to random, randomly select using
    # the hysimu_random_spectra_selector module
    # Import the module
    import hysimu_random_spectra_selector as hy_spec

    # HySIMU supports either "Mineral" or "Vegetation" types
    spectra_type = df.loc[
        df["parameter"] == "spectral_endmembers_material_type",
        "value"].iloc[0]

    # Randomly select the spectra
    endmembers = hy_spec.main(
        num_endmembers=num_endmembers,
        sensor_bands=sensor_bands,
        spectra_type=spectra_type,
        output_dir=project_dir,
        project_name=project_name
    )

# Invalid input spectral endmembers parameter
else:
    raise ValueError(
        "Input spectral endmembers invalid."
    )


# ======================================================================= #
# BUILD AND PROCESS SYNTHETIC SURFACE REFLECTANCE (GROUND TRUTH) DATACUBE
# ======================================================================= #
logger.info("Building surface reflectance datacube.")

# Check option to add spectral texture
add_spectral_texture = df.loc[
    df["parameter"] == "add_spectral_statistical_texture",
    "value"].iloc[0]

# If spectral texture is desired
if add_spectral_texture.lower() == "yes":
    logger.info("Adding spectral texture.")

    # Import the module
    import hysimu_spectral_texture as hy_texture

    # Read the list of spectral variance
    spectral_vars = df.loc[
        df["parameter"] == "spectral_variance_list",
        "value"].iloc[0]

    try:
        # Check if spectral variances are in a valid list
        spectral_vars = spectral_vars.split(',')
        spectral_vars = [float(var) for var in spectral_vars]
    except Exception:
        # If not, assumed default values
        spectral_vars = (0.05 * np.ones(num_endmembers)) ** 2
        logger.warning(
            "Invalid spectral variance. "
            "Spectral variance of 0.05^2 is assumed."
        )

    # Number of synthetic samples desired
    num_samples = df.loc[
        df["parameter"] == "num_of_synthetic_samples",
        "value"].iloc[0]

    # If the num_samples parameter is set as default
    if isinstance(num_samples, str):
        num_samples = 5
        logger.warning("5 synthetic samples for each spectra is assumed.")

    # Create the synthetic ground truth datacube & add texture if desired
    # using hysimu_spectral_texture module
    synthetic_ground_truth, statistical_spectra = hy_texture.main(
        subregion_map=subregion_map,
        num_row=num_row,
        num_col=num_col,
        sensor_bands=sensor_bands,
        endmembers=endmembers,
        num_endmembers=num_endmembers,
        num_subregions=num_subregions,
        spectral_vars=spectral_vars,
        num_samples=num_samples
    )

    # Save statistical spectra as a gzip file
    f = gzip.GzipFile(
        project_dir + "/" + project_name + "_statistical_spectra.npy.gz",
        "w")
    np.save(file=f, arr=statistical_spectra)
    f.close()

# Create ground truth datacube without texture
else:
    logger.info("No spectral texture is added.")

    # Initialize the array
    synthetic_ground_truth = np.zeros(
        (
            num_row,
            num_col,
            len(sensor_bands)
        ),
        dtype=np.float32
    )

    # Assign endmembers to each pixel based on the spectral map
    for i in range(num_row):
        for j in range(num_col):
            synthetic_ground_truth[i, j, :] = (
                endmembers[:, int(spectral_map[i, j] - 1)]
            )

# Check option to add control pixels, for sensitivity studies
control_pixels = df.loc[
    df["parameter"] == "add_control_pixels",
    "value"].iloc[0]

if control_pixels.lower() == "yes":
    logger.info(
        "Dark pixels in the top-left corner and "
        "bright pixels in the bottom-right corner."
    )
    for i in range(sensor_res):
        for j in range(sensor_res):
            # Dark pixels top left
            synthetic_ground_truth[i, j, :] = 0
            # Bright pixels bottom right
            synthetic_ground_truth[num_row - 1 - i, num_col - 1 - j, :] = 1
else:
    logger.info("No control pixels in the scene.")


# ======================================================================= #
# COMPUTE PIXEL-BASED SOLAR & VIEW GEOMETRY
# ======================================================================= #
# Datetime info
aq_date = df.loc[
    df["parameter"] == "acquisition_date",
    "value"].iloc[0]
aq_date = str(aq_date)
aq_time = df.loc[
    df["parameter"] == "acquisition_time",
    "value"].iloc[0]
aq_time = str(aq_time)
aq_datetime = " ".join([aq_date, aq_time])

# Coordinates of the reference point (pixel [0,0])
lat = df.loc[
    df["parameter"] == "top_left_pixel_latitude",
    "value"].iloc[0]
lon = df.loc[
    df["parameter"] == "top_left_pixel_longitude",
    "value"].iloc[0]

# Sensor geometry
view_az = df.loc[
    df["parameter"] == "view_azimuth_angle",
    "value"].iloc[0]
view_zen = df.loc[
    df["parameter"] == "view_zenith_angle",
    "value"].iloc[0]
sensor_altitude = df.loc[
    df["parameter"] == "sensor_altitude_in_m",
    "value"].iloc[0]

# Check option to calculate solar geomery
add_solar = df.loc[
    df["parameter"] == "compute_solar_geometry",
    "value"].iloc[0]

# DEM spatial resolution has to be the same as the ground truth
dem_res = gt_res

# Calculating solar and view geometry
if add_solar.lower() == "yes":
    logger.info("Computing solar and view angles.")

    # Import the module
    import hysimu_solar_geometry_computation as hy_solar

    # Compute/correct solar and view angles on non-flat DEM
    # using hysimu_solar_geomtery_calculation module
    sun_angles, view_angles, ref_sun_az, ref_sun_zen = hy_solar.main(
        latitude=lat,
        longitude=lon,
        datetime=aq_datetime,
        num_row=num_row,
        num_col=num_col,
        view_az=view_az,
        view_zen=view_zen,
        cell_size_m=dem_res,
        dem=dem_map
    )

# otherwise, calculation/correction is not needed
else:
    logger.info("Using reference solar and view angles.")

    # Import pvlib and timezone finder to get reference solar
    # position (pixel [0,0])
    from pvlib.solarposition import get_solarposition
    from timezonefinder import TimezoneFinder

    # Find timezone based on reference coordinates
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    # pvlib datetime format
    datetime_pvlib = pd.DatetimeIndex([aq_datetime], tz=tz_name)
    # Get solar position using pvlib
    sun_position = get_solarposition(datetime_pvlib, lat, lon)
    ref_sun_az = sun_position["azimuth"].iloc[0]
    ref_sun_zen = sun_position["zenith"].iloc[0]

    # All pixels have the same solar angles
    sun_angles = np.zeros_like(synthetic_ground_truth)
    sun_angles[:, :, 0] = ref_sun_az
    sun_angles[:, :, 0] = ref_sun_zen
    # All pixels have the same view angles
    view_angles = np.zeros_like(synthetic_ground_truth)
    view_angles[:, :, 0] = view_az
    view_angles[:, :, 0] = view_zen


# ======================================================================= #
# RADIATIVE TRANSFER MODEL
# ======================================================================= #
# RTM choice (6S / LRT / none)
rtm_choice = df.loc[
    df["parameter"] == "rtm_choice",
    "value"].iloc[0]

# ----------------------------------------------------------------------- #
# 6S RTM
# ----------------------------------------------------------------------- #
if rtm_choice.lower() == "6s":
    logger.info("Computing 6S RTM.")

    # Compute 6S RTM using hysimu_sixs_computation module

    # Import module
    import hysimu_sixs_computation as hy_sixs

    # 6S input parameters
    # 6S path
    sixs_path = df.loc[
        df["parameter"] == "6S_path",
        "value"].iloc[0]
    # 6S output type
    output_type = df.loc[
        df["parameter"] == "6S_output_type",
        "value"].iloc[0]

    # 6S atmospheric profile
    atmos = df.loc[
        df["parameter"] == "6S_atmosphere_profile",
        "value"].iloc[0]
    # 6S aerosol profile
    aero = df.loc[
        df["parameter"] == "6S_aerosol_profile",
        "value"].iloc[0]

    # Compute 6S
    rtm_datacube = hy_sixs.main(
        sixS_path=sixs_path,
        project_name=project_name,
        aero=aero,
        atmos=atmos,
        latitude=lat,
        date=aq_date,
        output_type=output_type,
        platform_type=platform_type,
        sensor_altitude=sensor_altitude,
        datacube=synthetic_ground_truth,
        sensor_wavelengths=sensor_bands,
        solar_angles_grid=sun_angles,
        view_angles_grid=view_angles,
        dem=dem_map,
        cores=cores,
        logger=logger,
        verbose_level=verbose_lvl
    )

# ----------------------------------------------------------------------- #
# LRT RTM
# ----------------------------------------------------------------------- #

elif rtm_choice.lower() == "lrt":
    logger.info("Computing LRT RTM.")

    # Compute libRadtran RTM using hysimu_lrt_computation module

    # Import module
    import hysimu_lrt_computation as hy_lrt

    # LRT input parameters
    # LRT path
    lrt_path = df.loc[
        df["parameter"] == "LRT_path",
        "value"].iloc[0]
    # Scratch directory to temporarily save lrt albedo files
    scratch_dir = df.loc[
        df["parameter"] == "LRT_scratch_directory_path",
        "value"].iloc[0]
    # LRT output type
    output_type = df.loc[
        df["parameter"] == "LRT_output_type",
        "value"].iloc[0]

    # LRT atmospheric profile
    atmos = df.loc[
        df["parameter"] == "LRT_atmosphere_profile",
        "value"].iloc[0]
    # LRT aerosol profile
    aero = df.loc[
        df["parameter"] == "LRT_aerosol_profile",
        "value"].iloc[0]

    # LRT H20 molar concentration in mm
    h2o_mm = df.loc[
        df["parameter"] == "LRT_H2O_mm",
        "value"].iloc[0]
    # LRT CO2 mixing ratio in ppm
    co2_ppm = df.loc[
        df["parameter"] == "LRT_CO2_ppm",
        "value"].iloc[0]
    # LRT CH4 mixing ratio in ppm
    ch4_ppm = df.loc[
        df["parameter"] == "LRT_CH4_ppm",
        "value"].iloc[0]

    # Compute LRT
    rtm_datacube = hy_lrt.main(
        lrt_path=lrt_path,
        output_type=output_type,
        scratch_directory=scratch_dir,
        project_name=project_name,
        datacube=synthetic_ground_truth,
        aero=aero,
        atmos=atmos,
        h2o_mm=h2o_mm,
        co2_ppm=co2_ppm,
        ch4_ppm=ch4_ppm,
        sensor_wavelengths=sensor_bands,
        platform_type=platform_type,
        sensor_altitude=sensor_altitude,
        datetime=aq_datetime,
        cores=cores,
        dem=dem_map,
        solar_angles_grid=sun_angles,
        view_angles_grid=view_angles,
        logger=logger,
        verbose_level=verbose_lvl
    )

# ----------------------------------------------------------------------- #
# NO RTM
# ----------------------------------------------------------------------- #
else:
    logger.warning(
        "No RTM applied.",
        stacklevel=2
    )

    output_type = df.loc[
        df["parameter"] == "no_rtm_output_type",
        "value"].iloc[0]

    if output_type.lower() == "radiance":

        # Compute radiance with no RTM using hysimu_radiance_computation
        # module
        # Import the module
        import hysimu_radiance_computation as hy_rad

        # Compute radiance
        rtm_datacube = hy_rad.main(
            datacube=synthetic_ground_truth,
            sensor_wavelengths=sensor_bands,
            solar_angles_grid=sun_angles,
            cores=cores,
            logger=logger,
            verbose_level=verbose_lvl
        )

    # Else if raw reflectance datacube is desired
    else:
        # Reflectance datacube
        rtm_datacube = synthetic_ground_truth


# ======================================================================= #
# PSF FILTER
# ======================================================================= #
# Check PSF filter option
psf_filter = df.loc[
    df["parameter"] == "PSF_filter_option",
    "value"].iloc[0]

# Ratio between higher resolution ground truth and lower resolution
# output datacube. Will be used to do convolution for spatial resampling
img_ratio = sensor_res / gt_res

# If SR2 PSF filter is selected
if psf_filter.lower() == "sr2":
    logger.info("Applying SR2 PSF filter.")

    # Compute the SR2 PSF spatial filter from a matlab function
    """
    Hyperspectral remote sensing Spatial Response Resampling (SR2)
        Point Spread Function (PSF) by Inamdar et al. 2023
        compiled as a python package called HSIBLUR
            [Inamdar, D., Kalacska, M., Darko, P. O.,
            Arroyo-Mora, J. P., & Leblanc, G. (2023).
            Spatial response resampling (SR2):
            Accounting for the spatial point spread function in
            hyperspectral image resampling. MethodsX, 10, 101998.
            https://doi.org/10.1016/j.mex.2023.101998]
    """
    # Import matlab functions as packages based on system type
    # Import the matlab module only after the MATLAB Compiler SDK
    # generated Python modules
    system_type = sys.platform
    if system_type == "win32":
        import HSIBLUR_WIN as HBlur
        import matlab
    elif system_type == "linux":
        import HSIBLUR_LNX as HBlur
        import matlab
    elif system_type == "darwin":
        import HSIBLUR_MAC as HBlur
        import matlab

    # Get all the input parameters
    flight_heading = df.loc[
        df["parameter"] == "flight_line_heading_in_degrees",
        "value"].iloc[0]
    FOV_deg = df.loc[
        df["parameter"] == "sensor_FOV_in_degrees",
        "value"].iloc[0]
    pix_tot = df.loc[
        df["parameter"] == "sensor_pixel_count",
        "value"].iloc[0]
    speed = df.loc[
        df["parameter"] == "platform_speed_in_meter_per_sec",
        "value"].iloc[0]
    it_time = df.loc[
        df["parameter"] == "sensor_integration_time_in_sec",
        "value"].iloc[0]
    cross_track_sum = df.loc[
        df["parameter"] == "cross_track_summing_factor",
        "value"].iloc[0]
    FWHM_opt = df.loc[
        df["parameter"] == "optical_FWHM",
        "value"].iloc[0]

    HSIBlur = HBlur.initialize()  # initialize function

    flight_line_heading = matlab.double([flight_heading], size=(1, 1))
    FOV_deg = matlab.double([FOV_deg], size=(1, 1))
    pix_tot = matlab.double([pix_tot], size=(1, 1))
    alt = matlab.double([sensor_altitude], size=(1, 1))
    speed = matlab.double([speed], size=(1, 1))
    it = matlab.double([it_time], size=(1, 1))
    cross_track_sum = matlab.double([cross_track_sum], size=(1, 1))
    FWHM_opt = matlab.double([FWHM_opt], size=(1, 1))
    pix_size_IMG = matlab.double([sensor_res], size=(1, 1))

    # Get the net_psf from the function
    PSF_tot_3d, net_psf = HSIBlur.HSI_BLUR(
        flight_line_heading, FOV_deg, pix_tot,
        alt, speed, it, cross_track_sum,
        FWHM_opt, pix_size_IMG, nargout=2
    )

    HSIBlur.terminate()  # terminate function

    net_psf = np.asarray(net_psf)

# Else if a simplified psf filter is selected
elif psf_filter.lower() == "simplified":
    logger.info(
        "Applying simplified PSF filter.",
        stacklevel=2
    )
    # Compute a simplified PSF filter using hysimu_psf_fitler_simplified
    # module

    # Import the module
    import hysimu_psf_filter_simplified as hy_psf

    # Gaussian PSF window size based on the assumption that more than
    # 50% pixel response originated from the surrounding pixels and
    # from references [Gonzalez and Woods, 2017, p. 168]
    # kernel radius ~ 3x pixel size (sigma): window size 6xsigma
    window_size = int(6 * img_ratio)
    convx = np.linspace(
        -int(np.floor(img_ratio)), int(np.ceil(img_ratio)), window_size
    )
    convy = np.linspace(
        -int(np.floor(img_ratio)), int(np.ceil(img_ratio)), window_size
    )
    convx, convy = np.meshgrid(convx, convy)
    conv_grid = (convx, convy)

    # Define parameters for PSFs
    gaussian_sigma = img_ratio
    # Gaussian PSF standard deviation
    rect_size = img_ratio  # size of rectangular PSF

    # net psf from convolution of a gaussian PSF and a rect PSF
    net_psf = hy_psf.main(
        grid=conv_grid,
        gaussian_sigma=gaussian_sigma,
        rect_size=rect_size
    )

# Otherwise, not PSF filter is assumed and the datacube will be spatially
# degraded only using a downsampling algorithm
else:
    logger.warning(
        "No PSF filter is applied.",
        stacklevel=2
    )


# ======================================================================= #
# SPATIAL MIXING FOR GROUND TRUTH DATACUBE
# ======================================================================= #
# Spatial mixing type
mix_order = df.loc[
    df["parameter"] == "spatial_mixing_order",
    "value"].iloc[0]

# Define output image size for each dimension
new_num_row = int(np.round((
    rtm_datacube.shape[0] * gt_res) / sensor_res
))
new_num_col = int(np.round((
    rtm_datacube.shape[1] * gt_res) / sensor_res
))
num_bands = rtm_datacube.shape[2]

# Initiate an empty spectrally-mixed datacube
new_datacube = np.zeros(
    (
        new_num_row,
        new_num_col,
        num_bands
    ),
    dtype=np.float32
)

# Get spatial mixing order parameter
mix_order = df.loc[
    df["parameter"] == "spatial_mixing_order",
    "value"].iloc[0]

# If either SR2 or simplified PSF is chosen
if psf_filter.lower() in ("sr2", "simplified"):
    logger.info("Computing spatial mixing with PSF.")

    # Spatially mixing, using a convolution to the net PSF,
    # and spatially resampling the RTM computed datacube using
    # hysimu_spatial mixing module

    # Import the module
    import hysimu_spatial_mixing as hy_mix

    # Spatial mixing and resampling using hy_mix
    output_datacube = hy_mix.main(
        datacube=rtm_datacube,
        num_col=new_num_col,
        num_row=new_num_row,
        new_datacube=new_datacube,
        net_psf=net_psf,
        mix_order=mix_order,
        cores=cores,
    )

# Otherwise, no PSF filter is applied but the datacube will
# still be spatially downsampled
else:
    logger.info(
        "Computing spatial mixing without PSF. "
        "Adding gaussian filter for anti-aliasing.")

    from skimage.transform import resize

    for i in range(new_datacube.shape[2]):
        new_datacube[:, :, i] = np.squeeze(
            resize(
                rtm_datacube[:, :, i],
                (new_num_row, new_num_col),
                order=mix_order,
                mode="reflect",
                anti_aliasing=True
            )
        )

    output_datacube = new_datacube


# ======================================================================= #
# OUTPUT FUNCTION
# ======================================================================= #
logger.info("Saving the output datacube to ENVI .bsq and .hdr.")

# Reference pixel for ENVI header is the top-left pixel
ref_pixelx = 1
ref_pixely = 1
# Sun positions for the reference pixel
ref_sun_elev = 90 - ref_sun_zen

# Create a metadata dict for ENVI header
metadata = {
    "description": (
        f"{project_name} {output_type} datacube "
        f"simulated based on {sensor_name} using {rtm_choice} RTM "
    ),
    "acquisition datetime": aq_datetime,
    "geo points": "{1, 1, " f"{lat}, {lon}""}",
    "pixel size": "{"f"{sensor_res}, {sensor_res}""}",
    "samples": new_num_col,
    "lines": new_num_row,
    "header offset": 0,
    "file type": "ENVI Standard",
    "data type": 4,
    "interleave": "bsq",
    "sensor type": sensor_name,
    "byte order": 0,
    "wavelength units": "nm",
    "sun azimuth": ref_sun_az,
    "sun elevation": ref_sun_elev,
    "bands": num_bands,
    "wavelength": "{"f'{str(list(sensor_bands))[1:-1]}'"}"
}

# save output datacube as a .BSQ ENVI file with a header
output_filename = project_dir + "/" + project_name + "_" + output_type + ".hdr"
spy.envi.save_image(
    output_filename, output_datacube,
    metadata=metadata, force=True,
    ext=".bsq", interleave='bsq', dtype=np.float32
)


# ======================================================================= #
logger.info("HySIMUv2.0 - END")
logger.info("--- Runtime: %.3f seconds ---" % (time.time() - start_time))


# ======================================================================= #
# ======================================================================= #
