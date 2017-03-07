## Crater lake energy and mass balance

This is a stochastic embedding of the model described by Hurst et al. [2014] to
compute energy and mass input required to account for observed temperature
changes in Mt. Ruapehu's crater lake. It has an interactive component which is currently implemented as a [Jupyter notebook](http://nbviewer.jupyter.org/github/ipython/ipython/blob/3.x/examples/Notebook/Index.ipynb).

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
First check out the source code:
```
git clone --depth=1 https://github.com/yannikbehr/clemb.git
```
Then compile the docker images:
```
cd clemb
docker build -t clemb .  
```
...and start the image:
```
docker run -it --rm -p 8888:8888 clemb
```
This will start a container from the docker image and expose the docker container's port 8888 to your local port 8888. It will also print a URL in the command line window to connect to the jupyter notebook. Copy and paste this into your browser. You should now see the start page of the jupyter notebook.

### Running the notebook
At this point you should see the start page of your jupyter notebook. Before starting the notebook `clemb_viz.ipynb` go to the `IPython Clusters` tab and start a cluster with the number of engines equal to the number of cores on your machine. Now go back to the `Files` tab, start the notebook, and play away.

#### *References*
Hurst, T., Hashimoto, T., & Terada, A. (2015). Crater Lake Energy and Mass Balance. In J. V. Dmitri Rouwet, Bruce Christenson, Franco Tassi (Ed.), Volcanic Lakes (pp. 1–533). Springer Berlin Heidelberg. http://doi.org/10.1007/978-3-642-36833-2_13
