# ======================================================================= #
# ======================================================================= #
"""
hysimu_random_map_generator
------
A function to create a random spectral map to simulate a
distribution of spectral zones on the surface.
The spectral zones are discretized from a fractal field generated using
inverse FFT of random noises normalized by Hurst-influenced
power spectrum synthesis coefficients of FFT frequencies.

This function follows, with adjustments, the algorithm described in
"Frequency Synthesis of Landscapes (and clouds)" by Paul Bourke
https://paulbourke.net/fractals/noise/
"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
import numpy as np
from scipy.ndimage import gaussian_filter


# ======================================================================= #
# FIELDS GENERATION FUNCTION
# ======================================================================= #
def generate_fields(
    num_col,
    num_row,
    num_regions,
    hurst,
    max_height,
    DEM_option,
    DEM_input,
    random_factor
):
    """
    Generate a 2D fractal noise field using power spectrum synthesis
    and inverse FFT

    Parameters:
        - num_col (int): 2nd dimension size of the field
        - num_row (int): 1nd dimension size of the field
        - num_regions (int): number of regions to which the fractal field will
                        be discretized. Corresponds to the number of
                        input spectral endmembers
        - hurst (float): hurst exponent coefficient. Determine the complexity
                    level of the fractal field. Higher ~ more complex
        - max_height (float): maximum altitude of the DEM, if generated
        - DEM_option (str): type of relation between DEM & spectral map
                                i) dem_based_fractal > fractal field
                                    generated from the input DEM
                                ii) input > DEM is a separate input to the
                                    distribution map
                                iii) random > DEM is generated from the same
                                    fractal field as the distribution map
        - DEM_input (array, float): DEM array, if assigned as an input
        - random_factor (float): controls the randomness level of the
                                    discretization of the fractal field.
                                    Needed if different discretization of
                                    the same DEM field desired. set to 0 if
                                    equal binning is desired

    Returns:
        - Fractal field (array, float): the raw fractal field
        - Region map (array, int): discretized fractal field of values
                                    ranging from 0 to the number of
                                    endmembers-1. these represent the
                                    "spectral zones"
        - DEM, if desired (array, float): DEM generated from the same fractal
                                            field
    """

    # If distribution map is based on the input DEM
    if DEM_option == "dem_based_fractal":
        # Input DEM is assumed to be the fractal field
        fractal_field = DEM_input

        # Set the dem variable to be empty, because it is already
        # an input and will not be prescribed by this function
        dem = []

    # Else, generate a fractal field and discretize
    else:
        # Create a grid of FFT frequencies
        kx, ky = np.meshgrid(np.fft.fftfreq(num_col), np.fft.fftfreq(num_row))
        k = np.sqrt(kx**2 + ky**2)
        k[0, 0] = 1  # Anticipate division by zero

        # Calculate power spectrum coeffs with Hurst exponent
        power_spectrum = k**(- (2 * hurst + 1))

        # Generate random Fourier noise components
        noise_real = np.random.normal(size=(num_row, num_col))
        noise_imag = np.random.normal(size=(num_row, num_col))
        # Normalize the noise using the power spectrum coeffs
        noise_ft = (noise_real + 1j * noise_imag) * np.sqrt(power_spectrum)

        # Inverse FFT to generate the spatial field
        fractal_field = np.real(np.fft.ifft2(noise_ft))

        # If DEM is already prescribed as an input, set the variable empty
        if DEM_option == "input":
            dem = []
        # Otherwise, the DEM will be generated from the same fractal field
        # to keep spatial correlation consistency, normalized by the
        # maximum altitude variable
        elif DEM_option == "random":
            dem = fractal_field * max_height
        else:
            dem = np.zeros_like(fractal_field)

    # Normalize the fractal field
    fractal_field -= fractal_field.min()
    fractal_field /= fractal_field.max()

    # Set the thresholds to discretize the fractal field
    thresholds = np.linspace(0, 1, num_regions + 1)
    # Add randomizer, so that the same DEM file can produce difference
    # discretization, if desired for sensitivity studies
    random_offsets = np.random.normal(
        0.5,
        random_factor,
        size=len(thresholds)
    )

    # Set the binnings from the thresholds
    bins = thresholds + random_offsets
    bins = np.sort(bins)  # Ensure bins remain sorted
    # Normaliziation
    bins -= bins.min()
    bins /= bins.max()

    # Discretize the fractal field
    region_map = np.digitize(fractal_field, bins) - 1
    region_map[region_map == num_regions] = num_regions - 1

    return fractal_field, region_map, dem


# ======================================================================= #
# SPATIAL TEXTURE FUNCTION
# ======================================================================= #
def add_spatial_texture(
    num_col,
    num_row,
    num_regions,
    num_subregions,
    region_map,
    spatial_smoothness
):
    """
    Add texture into map regions using randomized noise generation that
    can be smoothened. The noise will be populated by the synthesized
    spectra of each input endmember from the output of
    the hysimu_spectral_texture function

    Parameters:
        - num_col (int): 2nd dimension size of the field
        - num_row (int): 1nd dimension size of the field
        - num_regions (int): number of regions to which the fractal field will
                                be discretized. Corresponds to the number of
                                input spectral endmembers
        - num_subregions (int): the number of subregions that corresponds to
                                    the number of spectra synthetisized by the
                                    hysimu_spectral_texture function.
        - region_map (array): the region map generated by the above function
        - spatial_smoothness (int): the smoothness level of the
                                        "texture"/noise. Higher ~ smoother
        If no texture is desired, set num_subregions to 1

    Returns:
        - subregion_map (array, int): a map of textured region that
                                    represents the distribution of synthetic
                                    spectra
    """

    # Initialize the subregion map (region map with "textures"/noise)
    subregion_map = np.zeros_like(region_map)

    # Iterate over the regions
    for region in range(num_regions):
        # Get region mask & dimension
        mask = (region_map == region)
        shape = (num_row, num_col)

        # Generate random noise as texture
        noise = np.random.randn(*shape)
        # Smoothen the noise
        region_texture = gaussian_filter(
            noise, sigma=spatial_smoothness[region]
        )

        # Divide each region into subregions and fill with texture
        sub_thresholds = np.linspace(
            region_texture[mask].min(),
            region_texture[mask].max(),
            num_subregions
        )
        # Digitize the "subregions" to be populated with the synthetic
        # spectra from the hysimu_spectral_texture module
        XX = np.digitize(region_texture[mask], sub_thresholds)
        sub_region = XX - 1 + (num_subregions * region)

        subregion_map[mask] = sub_region

    return subregion_map


# ======================================================================= #
# MAIN FUNCTION
# ======================================================================= #
def main(
    num_row,
    num_col,
    complex_level,
    num_endmembers,
    num_subregions,
    max_height,
    DEM_option,
    DEM_input,
    random_factor,
    spatial_smoothness
):
    """
    Generate a spectral distribution map with texture

    Parameters:
        - Most are the same as previously described,
        - complex_level == Hurst
        - num_endmembers (int): number of endmembers in the scene.
                                corresponds to the number of spectral
                                regions/zones
    Returns:
        - Same as previously described
    """

    # Firstly, generate fractal field
    fractal_field, region_map, dem = generate_fields(
        num_col=num_col,
        num_row=num_row,
        num_regions=num_endmembers,
        hurst=complex_level,
        max_height=max_height,
        DEM_option=DEM_option,
        DEM_input=DEM_input,
        random_factor=random_factor
    )

    # Sceondly, add texture
    subregion_map = add_spatial_texture(
        num_col=num_col,
        num_row=num_row,
        num_regions=num_endmembers,
        num_subregions=num_subregions,
        region_map=region_map,
        spatial_smoothness=spatial_smoothness
    )

    return region_map, dem, subregion_map


# ======================================================================= #
# INITIALIZE MAIN FUNCTION
# ======================================================================= #
if __name__ == "__main__":
    main()


# ======================================================================= #
# ======================================================================= #
