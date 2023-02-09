from setuptools import setup

####################################################################
#                    CONFIGURATION
####################################################################

# do the build/install
setup(
    name="pumahu",
    version="0.1",
    description="Python package to compute crater lake energy and mass balance.",
    long_description="Python package to compute crater lake energy and mass balance.",
    author="Yannik Behr",
    author_email="yannikbehr@yanmail.de",
    url="",
    license="GPL v3",
    package_dir={'': 'src'},
    packages=['pumahu'],
    package_data={'pumahu': ['data/*.dat', 'data/*.csv', 'data/*.npz', 'notebook/*.ipynb']},
    entry_points={'console_scripts':
                  ['heat_mcmc=pumahu.mcmc:main',
                   'heat_uks=pumahu.uks:main',
                   'heat_dash=pumahu.dashboard:main',
                   'heat_api=pumahu.api:main']}
)
