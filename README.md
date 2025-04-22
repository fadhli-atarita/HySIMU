#  HySIMU
HYPERSPECTRAL REMOTE SENSING FORWARD MODELLING SIMULATOR

Author: Fadhli Atarita\
Year: 2025\
Version: 2.0

A Hyperspectral remote sensing forward modelling toolkit to create synthetic
hyperspectral at sensor reflectance or radiance datacubes based on:\
    - Sensor & platform parameters\
    - Mission parameters\
    - Atmospheric parameters (for Radiative Transfer Model)

Modules
-------------
HySIMU consists of the main "hysimu_main.py" script and several functions/modules, all located inside the hysimu directory ("/hysimu_dir"):

- __"hysimu_logging.py"__\
A function to centralize logging for hysimu modules. Except for joblib verbose.

- __"hysimu_random_map_generator.py"__\
A function to create a random spectral map to simulate a distribution of spectral zones (mineral or vegetation) on the surface.
The spectral zones are discretized from a fractal field generated using inverse FFT of random noises normalized by Hurst-influenced power spectrum synthesis coefficients of FFT frequencies.\
This function follows, with adjustments, the algorithm described in:
	> 		"Frequency Synthesis of Landscapes (and clouds)" by Paul Bourke https://paulbourke.net/fractals/noise/

- __"hysimu_random_spectra_selector.py"__\
A function to select random endmember spectra from the ECOSTRESS spectral library provided by spectral python.
	>	 	Meerdink, S. K., Hook, S. J., Roberts, D. A., & Abbott, E. A. (2019). The ECOSTRESS spectral library version 1.0.
	>		Remote Sensing of Environment, 230, 111196. https://doi.org/10.1016/j.rse.2019.05.015

- __"hysimu_spectral_texture.py"__\
A function to generate synthetic reflectance spectra of selected endmembers from user inputs or the hysimu_spectra_selector_function based on
covariance and standard deviation and populate the ground truth datacube with these spectra.\
This function follows, with adjustments, the algorithm described in:
	> 		Schott, J. R., Salvaggio, C., Brown, S. D., & Rose, R. A. (1995).
	> 		Incorporation of texture in multispectral synthetic image generation tools
	> 		W. R. Watkins & D. Clement, Eds.; pp. 189–196). https://doi.org/10.1117/12.210590

- __"hysimu_solar_geometry_computation.py"__\
A function to correct solar and view geometry on non-flat DEM, pixel by pixel. The outputs of this module will be used as inputs for RTM computation.
The calculation is based on the formula to compute the angle of incidence on an inclined surface:\
cos θ = cos θz cos β + sin θz sin β cos(γs − γ)
	>		Dufﬁe, J. A., & Beckman, W. A. (n.d.). Solar Engineering of Thermal Processes. p14

- __"hysimu_sixs_computation.py"__\
A function to parallelly compute 6S Radiative Transfer Model using the py6S wrapper.
	- 6S (Second Simulation of a Satellite Signal in the Solar) installation. Spectrum vector code by Vermote et al.
   		> https://salsa.umd.edu/6spage.html

			Vermote, E.F., Tanré, D., Deuzé, J.L., Herman, M., & Morcrette, J.-J. (1997), Second Simulation of the Satellite
   			Signal in the Solar Spectrum, 6S: An Overview, IEEE Transactions on Geoscience and Remote Sensing,
   			Vol. 35, No. 3, p. 675-686.

	  and its python wrapper py6S by Wilson
		> https://py6s.readthedocs.io/en/latest/

			Wilson, R. T. (2013). Py6S: A Python interface to the 6S radiative transfer model. Computers & Geosciences, 51, 166–171.
			ttps://doi.org/10.1016/j.cageo.2012.08.002

- __"hysimu_lrt_computation.py"__\
A function to parallelly compute libRadtran Radiative Transfer Model using the pyLRT wrapper.
	- RTM code by Mayer et al.
   		> http://www.libradtran.org/doku.php?id=start

			C. Emde, R. Buras-Schnell, A. Kylling, B. Mayer, J. Gasteiger, U. Hamann, J. Kylling, B. Richter, C. Pause, T. Dowling, and L. Bugliaro.
  			The libradtran software package for radiative transfer calculations (version 2.0.1).
  			Geoscientific Model Development, 9(5):1647-1672, 2016.

   			B. Mayer and A. Kylling. Technical note: The libRadtran software package for radiative transfer calculations - description and examples of use.
   			Atmos. Chem. Phys., 5: 1855-1877, 2005.

		and its python wrapper pyLRT by Gryspeerdt
  		> https://github.com/EdGrrr/pyLRT

- __"hysimu_radiance_computation.py"__\
A function to parallelly compute radiance without radiative transfer model based on the formula:\
Reflectance(λ) = π * TOA Radiance(λ) / Solar irradiance(λ) * cos(sun zenith)
	> 		Schowengerdt, R. A. (2007). Remote sensing, models, and methods for image processing (3rd ed). Academic Press. p52

- __"hysimu_psf_filter_simplified.py"__\
A function to compute a simplified PSF for hyperspectral imaging. A simplified PSF filter based on a gaussian distribution to simulate
an airy disk convolved with a rectangular response function image motion and electronic response are ignored.\
Calculation of the PSFs follows formulas provided in:
	> 		Schowengerdt, R. A. (2007). Remote sensing, models, and methods for image processing (3rd ed). Academic Press. p85-91

- __"hysimu_spatial_mixing.py"__\
A function to perform spatial mixing parallelly with joblib to downsample the surface reflectance datacube to sensor resolution,
using a previously computed net PSF filter.


Dependencies
-------------
- Python packages:
  	numpy, pandas, scipy, openpyxl, mat73, spectral python, py6S, pyLRT, joblib,
  	scikit-learn, pvlib, haversine, timezonefinder, insolation, matlabengine
    (including all of their respective dependencies).

- Matlab function:
Hyperspectral remote sensing Spatial Response Resampling (SR2)
  Point Spread Function (PSF) by Inamdar et al. 2023
  compiled as a python package called HSIBLUR.
  > Inamdar, D., Kalacska, M., Darko, P. O.,  Arroyo-Mora, J. P., & Leblanc, G. (2023).
    Spatial response resampling (SR2): Accounting for the spatial point spread function in
    hyperspectral image resampling. MethodsX, 10, 101998. https://doi.org/10.1016/j.mex.2023.101998.

    This matlab function depends on the platform on which hysimu is run. See suffixes:
  - _LNX for linux
  - _WIN for windows
  - _MAC for macos (not included yet)

  It requires either a full matlab OR the freely available matlab runtime installation version 2024b.

  The package can be found inside the "hysimu_dependencies" directory and
        steps to install these packages as python packages can be found on:
  > [https://www.mathworks.com/help/compiler_sdk/python/import-compiled-python-packages.html]

- Radiative Transfer Models:
  - 6S (Second Simulation of a Satellite Signal in the Solar) installation.
        Spectrum vector code and its python wrapper py6S.\
            [see references list above]

  - libRadtran (library for radiative transfer) installation.
        RTM code and its python wrapper pyLRT.\
            [see references list above]

    These RTM solvers have to be installed on the machine. Can be either
    or both, depending on the application.


Inputs
-------
Templates can be found inside the "hysimu_files" directory.

- An ecostress spectral library compiled by the spectral python package
    called 'ecostress.db' located inside '/hysimu_dir'.
- Input parameters should be written in an .xlsx/.csv file (see template).
- A SLURM batch file (see template).

Usage
------
Add all input files inside the project directory and run the batch file
    directly from the project directory. keep '/hysimu_dir' separate.

Outputs
--------
- A spectral distribution map (if "random") as a compressed npy.gzip file.
- A spectral endmember spectra, names and ids (if "random") or a resampled
	spectra to the output/sensor resolution (if spectra endmembers are an
	input file).
- A statistical spectra file which contains all generated spectra (if spectral
	texture is set to "yes").
- A DEM file (if "random") as a compressed npy.gzip file.
- An output datacube (radiance or reflectance) as a .BSQ ENVI file along with
	a header .hdr file
	Radiance datacube units:
		> with 6S: watt / meter-squared steradian micrometer
		> with LRT: miliwatt / meter-squared steradian nanometer
		> with no RTM : watt / meter-squared steradian micrometer

License
--------
GNU General Public License v3.0 (GNU GPLv3)
