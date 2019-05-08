from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup

####################################################################
#                    CONFIGURATION
####################################################################

# do the build/install
setup(
    name="clemb",
    version="0.0.3",
    description="Python package to compute crater lake energy and mass balance.",
    long_description="Python package to compute crater lake energy and mass balance.",
    author="Yannik Behr",
    author_email="yannikbehr@yanmail.de",
    url="",
    license="GPL v3",
    package_dir={'': 'src'},
<<<<<<< HEAD
    install_requires=['pandas<=0.22', 'numpy<=1.14', 'progressbar2', 'filterpy',
                      'scipy<=1.0', 'xarray<=0.10'],
=======
    install_requires=['pandas<=0.22.*', 'numpy<=1.14', 'progressbar2', 'filterpy',
                      'scipy<=1.0*', 'xarray<=0.10.*'],
>>>>>>> 08a645cda03d9ae64bc8f6e829fd8b600d8df41e
    packages=['clemb'],
    package_data={'clemb': ['data/*.dat', 'data/*.npz', 'notebook/*.ipynb']}
)
