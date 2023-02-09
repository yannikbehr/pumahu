## Pumahu (=steam/heat in [Te Reo Maori](https://maoridictionary.co.nz/search?idiom=&phrase=&proverb=&loan=&histLoanWords=&keywords=pumahu) )

This package implements a non-linear Kalman Smoother to make continuous estimates of the heat flow into Ruapehu Crater Lake (Te Wai a-moe), New Zealand. For details of the method see [Behr et al. [2023]](https://rdcu.be/c47TL).

### Installation

#### Local
First check out the source code:
```
git clone --depth=1 https://github.com/yannikbehr/pumahu.git
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
