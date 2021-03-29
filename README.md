## Crater lake energy and mass balance

This is a stochastic embedding of the model described by Hurst et al. [2014] to
compute energy and mass input required to account for observed temperature
changes in Mt. Ruapehu's crater lake.

### Installation

#### Local
##### Requirements:
* python >= 3.5
* pandas >= 0.19
* numpy >= 1.11
* scipy >= 0.18
* bokeh >= 0.12
* ipyparallel >= 6.1
* ipywidgets >= 6.0

First check out the source code:
```
git clone --depth=1 https://github.com/yannikbehr/clemb.git
```

Then install the python package:
```
cd clemb
python setup.py install
```
or if administrator permissions are required

```
cd clemb
sudo python setup.py install
```
If you don't have the `IPython Clusters` tab already enabled run:
```
ipcluster nbextension enable
```
Now start the notebook:
```
cd notebook
jupyter notebook&
```

#### Docker
First pull the image from docker-hub:
```
git pull yadabe/clemb:latest
```
Then start the image:
```
docker run -it --rm -p 8888:8888 yadabe/clemb
```
### Contribute

Documentation should follow the numpy documentation style:
https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

Issue Tracker: https://git.gns.cri.nz/behrya/pumahu/-/issues
Source Code: https://git.gns.cri.nz/behrya/pumahu

### Running the notebook
At this point you should see the start page of your jupyter notebook. Before starting the notebook `clemb_viz.ipynb` go to the `IPython Clusters` tab and start a cluster with the number of engines less or equal to the number of cores on your machine. Now go back to the `Files` tab, start the notebook, and click `Cell->Run All`. This will execute the notebook. To see the interactive part of the notebook scroll to the bottom. Note that you will have to confirm entries in the interactive text fields with `Enter`.

#### *References*
Hurst, T., Hashimoto, T., & Terada, A. (2015). Crater Lake Energy and Mass Balance. In J. V. Dmitri Rouwet, Bruce Christenson, Franco Tassi (Ed.), Volcanic Lakes (pp. 1–533). Springer Berlin Heidelberg. http://doi.org/10.1007/978-3-642-36833-2_13
