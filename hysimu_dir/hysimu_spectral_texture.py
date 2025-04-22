# ======================================================================= #
"""
hysimu_spectral_texture
------
A function to generate synthetic reflectance spectra of selected endmembers
from user inputs or the hysimu_spectra_selector_function based on
covariance and standard deviation and populate the ground truth datacube
with them

This function follows, with adjustments, the algorithm described in
Schott, J. R., Salvaggio, C., Brown, S. D., & Rose, R. A. (1995).
Incorporation of texture in multispectral synthetic image generation tools
(W. R. Watkins & D. Clement, Eds.; pp. 189–196).
https://doi.org/10.1117/12.210590
"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
import numpy as np
from scipy.stats import multivariate_normal


# ======================================================================= #
# VALID SPECTRA FILTER FUNCTION
# ======================================================================= #
def valid_spectra_filter(
    mean,
    cov,
    num_spectra
):
    """
    Check the generated spectra and filter out if any of the spectra has
    values lower than 0 or higher than 1

    Parameters:
        - mean (list, float): the main/base reflectance spectra
        - cov (array, float): 1nd dimension size of the field
        - num_spectra (int): the desired number of synthetic spectra + the
                                original

    Returns:
        - generated_spectra (array, float): an array of the generated
                                            synthetic spectra.
                                            row: num of spectra;
                                            col: num of bands
    """

    # Initialized output list
    generated_spectra = []
    # Condition while the "valid" generated spectra are still less than
    # the desired number
    while len(generated_spectra) < num_spectra:
        # Generate a batch with multivariate normal distribution
        spectra_batch = multivariate_normal.rvs(
            mean=mean, cov=cov, size=num_spectra
        )

        # Keep only spectra where all values are in [0, 1]
        if spectra_batch.ndim == 1:
            # Ensure 2D if batch is only 1
            spectra_batch = spectra_batch[np.newaxis, :]

        # Check the values of the spectra and register it as valid
        valid_spectra = spectra_batch[
            (spectra_batch >= 0).all(axis=1) & (spectra_batch <= 1).all(axis=1)
        ]

        # Append valid spectra
        generated_spectra.extend(valid_spectra)

    return np.array(generated_spectra[:num_spectra])  # Change to np array


# ======================================================================= #
# MAIN FUNCTION
# ======================================================================= #
def main(
    subregion_map,
    num_row,
    num_col,
    sensor_bands,
    endmembers,
    num_endmembers,
    num_subregions,
    spectral_vars,
    num_samples
):
    """
    Populate the previously generated, spatially-textured spectral zones
    map with synthetic spectra created using statistical distributiuon
    algorithm

    Parameters:
        - subregion_map (array, int): an array of the discretized spectral
                                        zones/distribution map
        - num_col (int): 2nd dimension size of the subregion_map
        - num_row (int): 1nd dimension size of the subregion_map
        - sensor_bands (array, float): output/sensor band wavelengths
                                        1D array
        - endmembers (array, float): an array of the input or selected
                                        endmembers.
                                        rows: bands; cols: types
        - num_endmembers (int): number of endmembers in the scene.
                                corresponds to the number of spectral
                                regions/zoness
        - num_subregions (int): the number of subregions that corresponds to
                                    the number of spectra synthetisized/
                                    spectral texture
        - spectral_vars (list, float): a list of variances that will be used
                                        to generate the synthetic spectral
                                        textures
        - num_samples (int): the number of the initial sample of synthetic
                                spectra to calcualte the covariance matrix
                                from
        If no texture is desired, set all spectral_vars as 0

    Returns:
            - textured datacube (array, float): a surface reflectance datacube
                                                spectral and spatial texture
            - statistical_spectra (array, float): an array of the generated
                                                    synthetic spectra
    """

    # Initialize the output datacube
    textured_datacube = np.zeros(
        (
            num_row,
            num_col,
            len(sensor_bands)
        ),
        dtype=np.float32
    )

    # Initialize statistical_spectra output list
    statistical_spectra = []

    # Count subregions from the subregion map
    unique, counts = np.unique(subregion_map, return_counts=True)

    # Loop through all endmembers and distribute them
    for val in range(num_endmembers):
        # Use the original endmember spectrum as base/mean
        base_reflectance = endmembers[:, val]

        # Calculate std from input variances
        std_est = np.sqrt(spectral_vars[val]) * np.abs(base_reflectance)
        # Synthetic initial samples to build the cov matrix from
        generated_samples = [
            base_reflectance + std_est * shift
            for shift in np.linspace(-1, 1, num_samples - 1)
        ]

        # Stack all samples into an array
        data_matrix = np.vstack([base_reflectance] + generated_samples)

        # Compute the covariance matrix and ensure positive semi-definiteness
        cov_matrix = (
            np.cov(data_matrix, rowvar=False)
            + np.eye(len(sensor_bands))
            * 1e-6
        )

        # Generate synthetic spectra as many as the required subregions
        num_spectra = num_subregions
        generated_spectra = valid_spectra_filter(
            mean=base_reflectance,
            cov=cov_matrix,
            num_spectra=num_spectra
        )

        statistical_spectra.append(generated_spectra)

        # Find index locations of each subregions and populated each
        # with the synthetic spectra
        p = 0
        for i in range((val * num_spectra), ((val + 1) * num_spectra)):
            idx = np.argwhere(subregion_map == i)  # subregions index
            for k in range(len(idx)):
                # Find pixels by the index and based on the index
                # populate the pixels with synthetic spectra
                textured_datacube[idx[k, 0], idx[k, 1], :] = (
                    generated_spectra[p, :]
                )
            p += 1

    return textured_datacube, statistical_spectra


# ======================================================================= #
# INITIALIZE MAIN FUNCTION
# ======================================================================= #
if __name__ == "__main__":
    main()


# ======================================================================= #
# ======================================================================= #
