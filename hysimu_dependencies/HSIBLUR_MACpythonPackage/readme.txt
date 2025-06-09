HSIBLUR_MAC MATLAB Python Package

1. Prerequisites for Deployment 

Verify that MATLAB Runtime(R2024b) is installed.   
If not, you can run the MATLAB Runtime installer.
To find its location, enter
  
    >>mcrinstaller
      
at the MATLAB prompt.
NOTE: You will need administrator rights to run the MATLAB Runtime installer. 

Alternatively, download and install the Macintosh version of the MATLAB Runtime for R2024b 
from the following link on the MathWorks website:

    https://www.mathworks.com/products/compiler/mcr/index.html
   
For more information about the MATLAB Runtime and the MATLAB Runtime installer, see 
"Distribute Applications" in the MATLAB Compiler SDK documentation  
in the MathWorks Documentation Center.

Verify that a Macintosh version of Python 3.9, 3.10, 3.11, and/or 3.12 is installed.

2. Installing the HSIBLUR_MAC Package

A. Change to the directory that contains the file setup.py and the subdirectory 
HSIBLUR_MAC. If you do not have write permissions, copy all its contents to a temporary 
location and change to that directory.

B. Execute the command:

    python -m pip install . [options]
   
If you have full administrator privileges, and install to the default location, you do 
not need to specify any options. Otherwise, use --user to install to your home folder, or 
--prefix="installdir" to install to "installdir". In the latter case, add "installdir" to 
the PYTHONPATH environment variable. For details, refer to:

    https://docs.python.org/3/installing/index.html

C. Set environment variables as follows:

In the following directions, replace MR/R2024b by the directory on the target machine where MATLAB is installed, or MR by the directory where the MATLAB Runtime is installed.

If the environment variable DYLD_LIBRARY_PATH is undefined, set it to the following string:

MR/R2024b/runtime/maca64:MR/R2024b/sys/os/maca64:MR/R2024b/bin/maca64

If it is defined, set it to the following:

${DYLD_LIBRARY_PATH}:MR/R2024b/runtime/maca64:MR/R2024b/sys/os/maca64:MR/R2024b/bin/maca64

3. Using the HSIBLUR_MAC Package

On the Mac, you must use the mwpython script, found in the bin directory of your MATLAB 
Runtime, to start a session or script that imports your package. Execute:

    mwpython -help
    
for more details. 

NOTE: Ensure that you are using a 64-bit JVM.
The HSIBLUR_MAC package is on your Python path. To import it into a Python script or 
session, execute:

    import HSIBLUR_MAC

If a namespace must be specified for the package, modify the import statement accordingly.
