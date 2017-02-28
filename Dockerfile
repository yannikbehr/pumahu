FROM jupyter/base-notebook

MAINTAINER Yannik Behr

USER root
RUN apt-get update && \
    apt-get install -y \
    git \
    openssh-server \
    && apt-get clean

USER $NB_USER
# Install Python 3 packages
RUN conda install --quiet --yes \
    'ipywidgets=5.2*' \
    'pandas=0.19*' \
    'scipy=0.18*' \
    'bokeh=0.12*' && \
    'ipyparallel=6.*' \
    conda clean -tipsy

# Activate ipywidgets extension in the environment that runs the notebook server
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix

USER root
WORKDIR /usr/local/src
RUN git clone https://github.com/yannikbehr/clemb.git
WORKDIR /usr/local/src/clemb
RUN python3 setup.py install

USER $NB_USER

CMD ["/bin/bash", "/usr/local/src/clemb/start_docker.sh"]

