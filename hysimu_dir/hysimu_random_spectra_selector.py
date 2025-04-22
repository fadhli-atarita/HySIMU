# ======================================================================= #
"""
hysimu_random_spectra_selector
------
A function to select random endmember spectra from the ECOSTRESS spectral
library provided by spectral python.

ECOSTRESS library:
    [Meerdink, S. K., Hook, S. J., Roberts, D. A., & Abbott,
    E. A. (2019). The ECOSTRESS spectral library version 1.0.
    Remote Sensing of Environment, 230, 111196.
    https://doi.org/10.1016/j.rse.2019.05.015]
"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
import spectral as spy
import numpy as np
import random
from collections import defaultdict
import os


# ======================================================================= #
# MAIN FUNCTION
# ======================================================================= #
def main(
    num_endmembers,
    sensor_bands,
    spectra_type,
    output_dir,
    project_name
):
    """
    Select random material spectra from spectral library database
    and resample them to sensor bands

    Parameters:
        - num_endmembers (int): number of endmembers in the scene.
                                corresponds to the number of spectral
                                regions/zones
        - sensor_bands (array, float): output/sensor band wavelengths
                                        1D array
        - spectra_type (str): materials type to be selected. "mineral"
                                or "vegetation"
        - output_dir (str): output directory to save an array of selected
                            endmember spectra
        - project_name (str): project name as output filename identifier

    Returns:
        - endmembers (array): 2D array with the rows being the
            index of the sensor bands and columns being the number of spectra
            selected
    """

    # Import ecostress database from HySIMU directory
    db_path = os.path.dirname(os.path.realpath(__file__))
    db = spy.EcostressDatabase(db_path + '/ecostress.db')

    # Set input bands
    sensor_bands = sensor_bands * 0.001  # convert to microns
    # smin = np.min(sensor_bands)  # saved for future options
    # smax = np.max(sensor_bands)  # saved for future options

    # Query the database
    # -------------- #
    # Initialize endmembers array
    endmembers = np.zeros(
        (len(sensor_bands), num_endmembers), dtype=np.float32
    )

    # Create the database query, based on material type and
    # wavelengths
    sql = (
        "SELECT SpectrumID, Name FROM Spectra, Samples "
        "WHERE Samples.SampleID = Spectra.SampleID AND "
        f"Type LIKE '%{spectra_type}%' AND "
        f"MinWavelength <= 0.4 "
        f"AND MaxWavelength >= 2.5"
        # f"MinWavelength <= {smin} "  # saved for future options
        # f"AND MaxWavelength >= {smax}"  # saved for future options
    )

    # Search the database based on the query
    a = db.query(sql)
    data = [(r[0], r[1]) for r in a]  # extract id and name

    # Group the same material and use unique spectra
    # Group ids by number
    number_to_ids = defaultdict(list)
    for id_, number in data:
        number_to_ids[number].append(id_)

    # Select unique endmembers
    if num_endmembers > len(number_to_ids):
        raise ValueError("Not enough unique numbers to sample from.")

    unique_numbers = random.sample(list(number_to_ids.keys()), num_endmembers)

    # Select one id for each unique endmembers
    selected_pairs = [
        (id_, number)
        for number in unique_numbers
        for id_ in random.sample(number_to_ids[number], 1)
    ]

    # Save pairs of ids and name for all selected endmembers as a file
    with open(
        output_dir + "/" + project_name + "_endmembers_ids.txt", "w"
    ) as f:
        for t in selected_pairs:
            f.write(' '.join(str(s) for s in t) + "\n")

    # Resample endmembers to sensor wavelengths
    k = 0
    for ids, names in selected_pairs:
        spectrum = db.get_signature(ids)
        resampler = spy.BandResampler(spectrum.x, sensor_bands)
        # Normalize values to 0-1. ecostress uses 0-100.
        endmembers[:, k] = resampler(spectrum.y) * 0.01
        k += 1

    # Save spectra. Columns = number of endmembers, Rows = bands index
    np.savetxt(
        output_dir + "/" + project_name + "_endmembers_spectra_resampled.dat",
        endmembers
    )

    return endmembers


# ======================================================================= #
# INITIALIZE MAIN FUNCTION
# ======================================================================= #
if __name__ == "__main__":
    main()


# ======================================================================= #
# ======================================================================= #
