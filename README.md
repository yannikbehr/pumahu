## Crater lake energy and mass balance

A package that contains different inversion algorithms to estimate the heat and
steam input into Mt. Ruapehu's crater lake using the physical model described
by Hurst et al. [2014] and Stevenson [1992].

### Installation

#### Local
First check out the source code:
```
git clone --depth=1 https://tanuki.gns.cri.nz/behrya/pumahu.git
```

Setup the environment:
```
cd pumahu
conda env create -f environment.yml
```

Then install the python package:
```
conda activate pumahu
python setup.py install
```

#### Docker
To get an interactive prompt inside a container, run:

```
./buildnrun.sh -b -i
```

### Contribute

Documentation should follow the numpy documentation style:
https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

Issue Tracker: https://git.gns.cri.nz/behrya/pumahu/-/issues
Source Code: https://git.gns.cri.nz/behrya/pumahu

#### *References*
Hurst, T., Hashimoto, T., & Terada, A. (2015). Crater Lake Energy and Mass Balance. In J. V. Dmitri Rouwet, Bruce Christenson, Franco Tassi (Ed.), Volcanic Lakes (pp. 1–533). Springer Berlin Heidelberg. http://doi.org/10.1007/978-3-642-36833-2_13
D. S. Stevenson (1992), “Heat transfer in active volcanoes: models of crater lake systems,” The Open University
