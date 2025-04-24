# ======================================================================= #
"""
hysimu_solar_geometry_computation
------
A function to correct solar and view geometry on non-flat DEM,
pixel by pixel. The outputs of this module will be used as inputs for
RTM computation.

The calculation is based on the formula to compute the angle of incidence
on an inclined surface:
            cos θ = cos θz cos β + sin θz sin β cos(γs − γ)
[Dufﬁe, J. A., & Beckman, W. A. (n.d.).
Solar Engineering of Thermal Processes. p14]
"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
import numpy as np
from insolation import insolf
from pvlib.solarposition import get_solarposition
import pandas as pd
from timezonefinder import TimezoneFinder
from haversine import inverse_haversine, Direction


# ======================================================================= #
# MAIN FUNCTION
# ======================================================================= #
def main(
    latitude,
    longitude,
    datetime,
    num_row,
    num_col,
    view_az,
    view_zen,
    cell_size_m,
    dem
):
    """
    Compute DEM-based solar and view geometry. Reference pixel is
    [0, 0], top left

    Parameters:
        - latitude (float): latitude of reference point
        - longitude (float): longitude of reference point
        - datetime (str): a datetime string
        - num_col (int): 2nd dimension size of the field
        - num_row (int): 1nd dimension size of the field
        - view_az (float): sensor view azimuth angle
        - view_zen (float): sensor view zenith angle
        - cell_size_m (float): size of the dem pixel in meter
        - dem (array, float): DEM array

    Returns:
        - sun_corr_grid (array, float): corrected solar angles grid
                                        (dim 3, index 0 is azimuth,
                                        1 zenith)
        - view_corr_grid (array, float) : corrected view angles grid
                                            (dim3, index 0 is azimuth,
                                            1 zenith)
        - ref_sun_az (float): sun azimuth at the reference pixel
        - ref_sun_zen(float): sun zenith at the reference pixel
    """

    # Set location parameters
    # Find timezone
    tf = TimezoneFinder()  # set timezone
    tz_name = tf.timezone_at(lat=latitude, lng=longitude)
    # Set pvlib datetime parameter
    datetime_pvlib = pd.DatetimeIndex([datetime], tz=tz_name)

    # Coordinates
    loc = (latitude, longitude)

    # Firstly, create an array of coordinates for every pixel
    # using haversine library and the reference coordinates as
    # base

    # Create a distance to reference pixel grid
    col_dist = np.arange(num_col) * cell_size_m * 0.001  # change to km
    row_dist = np.arange(num_row) * cell_size_m * 0.001

    # Column coordinates
    col_coord = [
        inverse_haversine(loc, col_dist[col], Direction.EAST)
        for col in range(num_col)
    ]  # array calculation is eastward from reference pixel [0,0]
    # Row coordinates
    row_coord = [
        inverse_haversine(loc, row_dist[row], Direction.SOUTH)
        for row in range(num_row)
    ]  # array calculation is southward from reference pixel [0,0]

    # lon and lat arrays
    col_lon = [lon for _, lon in col_coord]
    row_lat = [lat for lat, _ in row_coord]

    # Construct the corrected coordinate grid
    col_coord_grid, row_coord_grid = np.meshgrid(col_lon, row_lat)

    # Secondly, calculate sun positions at every pixel
    # Initialize sun and view grids
    sun_grid = np.zeros((num_row, num_col, 2), dtype=np.float32)
    view_grid = np.zeros_like(sun_grid)
    # Set index 0 as azimuth and index 1 as zenith!!
    view_grid[:, :, 0] = view_az
    view_grid[:, :, 1] = view_zen

    # Calculate sun positons using pvlib
    for row in range(num_row):
        for col in range(num_col):
            lonx = col_coord_grid[row, col]
            latx = row_coord_grid[row, col]
            sun_position = get_solarposition(datetime_pvlib, latx, lonx)
            sun_az = sun_position["azimuth"].iloc[0]
            sun_zen = sun_position["zenith"].iloc[0]

            sun_grid[row, col, 0] = sun_az
            sun_grid[row, col, 1] = sun_zen

    # Set reference pixel sun position
    ref_sun_az = sun_grid[0, 0, 0]
    ref_sun_zen = sun_grid[0, 0, 1]

    # Thirdly, calculate sun and view incidence angle at every pixel
    # Calculate DEM slope and aspect for every pixel
    demslope = insolf.slope(dem, cell_size_m, degrees=True)
    demaspect = insolf.aspect(dem, cell_size_m, degrees=True)

    # Angle normalization
    # Hysimu convention: sun az (0 deg in N; 90 deg in E; 180 deg in S)
    # Insolation & pvlib convention is the same
    # However, the formula from Duffie & Beckman has a different convention
    # 0 deg sun in S; +90 sun in W; -90 sun in E; +-180 sun in N
    # normalize both aspect and sun az
    demaspect -= 180
    sun_grid[:, :, 0] -= 180

    # Initialize corrected arrays
    sun_corr_grid = np.zeros_like(sun_grid)
    view_corr_grid = np.zeros_like(view_grid)

    # Calculate incident solar and view angles
    # cos θ = cos θz cos β + sin θz sin β cos(γs − γ)
    for i in range(num_row):
        for j in range(num_col):
            # Solar angles
            sun_inc = (
                np.cos(np.deg2rad(sun_grid[i, j, 1]))
                * np.cos(np.deg2rad(demslope[i, j]))
                + np.sin(np.deg2rad(sun_grid[i, j, 1]))
                * np.sin(np.deg2rad(demslope[i, j]))
                * np.cos(
                    np.deg2rad(
                        (sun_grid[i, j, 0]) - np.deg2rad(demaspect[i, j])
                    )
                )
            )

            sun_corr_grid[i, j, 1] = np.rad2deg(np.arccos(sun_inc))
            # Normalize back to hysimu convention
            sun_corr_grid[i, j, 0] = sun_grid[i, j, 0] + 180

            # View angles
            view_inc = (
                np.cos(np.deg2rad(view_grid[i, j, 1]))
                * np.cos(np.deg2rad(demslope[i, j]))
                + np.sin(np.deg2rad(view_grid[i, j, 1]))
                * np.sin(np.deg2rad(demslope[i, j]))
                * np.cos(
                    np.deg2rad(
                        (view_grid[i, j, 0]) - np.deg2rad(demaspect[i, j])
                    )
                )
            )

            view_corr_grid[i, j, 1] = np.rad2deg(np.arccos(view_inc))
            # Normalize back to hysimu convention
            view_corr_grid[:, :, 0] = view_grid[i, j, 0] + 180

    # Reminder: index 0 is azimuth and index 1 is zenith!!

    return sun_corr_grid, view_corr_grid, ref_sun_az, ref_sun_zen


# ======================================================================= #
# INITIALIZE MAIN FUNCTION
# ======================================================================= #
if __name__ == "__main__":
    main()


# ======================================================================= #
# ======================================================================= #
