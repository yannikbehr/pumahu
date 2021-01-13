from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup

####################################################################
#                    CONFIGURATION
####################################################################

# do the build/install
setup(
    name="pumahu",
    version="0.0.4",
    description="Python package to compute crater lake energy and mass balance.",
    long_description="Python package to compute crater lake energy and mass balance.",
    author="Yannik Behr",
    author_email="yannikbehr@yanmail.de",
    url="",
    license="GPL v3",
    package_dir={'': 'src'},
    packages=['pumahu'],
    package_data={'pumahu': ['data/*.dat', 'data/*.npz', 'notebook/*.ipynb']},
    entry_points={'console_scripts':
                  ['heat_mcmc=pumahu.mcmc:main']}
)
