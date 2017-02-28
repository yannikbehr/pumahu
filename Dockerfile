FROM jupyter/base-notebook

MAINTAINER Yannik Behr

USER root
RUN apt-get update && \
    apt-get install -y \
    git \
    openssh-server \
    && apt-get clean

# Grant NB_USER permission to /usr/local
RUN sudo chgrp -R users /usr/local
RUN sudo chmod -R g+w /usr/local

USER $NB_USER
# Install Python 3 packages
RUN conda install --quiet --yes \
    'ipywidgets=5.2*' \
    'pandas=0.19*' \
    'scipy=0.18*' \
    'bokeh=0.12*' \
    'ipyparallel' \
    && conda clean -tipsy

# Activate ipywidgets extension in the environment that runs the notebook server
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix
RUN ipcluster nbextension enable

WORKDIR /usr/local/src
RUN git clone --depth=1 https://github.com/yannikbehr/clemb.git
WORKDIR /usr/local/src/clemb
RUN python3 setup.py install

CMD jupyter notebook --no-browser --ip=* --notebook-dir=/usr/local/src/clemb/notebook --port=8888

