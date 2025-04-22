#!/usr/bin/env python
"""
Sample script that uses the HSIBLUR_LNX package created using
MATLAB Compiler SDK.

Refer to the MATLAB Compiler SDK documentation for more information.
"""

import HSIBLUR_LNX
# Import the matlab module only after you have imported 
# MATLAB Compiler SDK generated Python modules.
import matlab

try:
    my_HSIBLUR_LNX = HSIBLUR_LNX.initialize()
except Exception as e:
    print('Error initializing HSIBLUR_LNX package\\n:{}'.format(e))
    exit(1)

try:
    flight_line_headingIn = matlab.double([90], size=(1, 1))
    FOV_degIn = matlab.double([30], size=(1, 1))
    pix_totIn = matlab.double([1920], size=(1, 1))
    altIn = matlab.double([2000], size=(1, 1))
    speedIn = matlab.double([100], size=(1, 1))
    itIn = matlab.double([0.009], size=(1, 1))
    cross_track_sumIn = matlab.double([1], size=(1, 1))
    FWHM_optIn = matlab.double([1], size=(1, 1))
    pix_size_IMGIn = matlab.double([0.1], size=(1, 1))
    PSF_tot_3dOut, conv_kerOut = my_HSIBLUR_LNX.HSI_BLUR(flight_line_headingIn, FOV_degIn, pix_totIn, altIn, speedIn, itIn, cross_track_sumIn, FWHM_optIn, pix_size_IMGIn, nargout=2)
    print(PSF_tot_3dOut, conv_kerOut, sep='\\n')
except Exception as e:
    print('Error occurred during program execution\\n:{}'.format(e))

my_HSIBLUR_LNX.terminate()