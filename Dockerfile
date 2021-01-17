FROM python:3.7 

MAINTAINER Yannik Behr <y.behr@gns.cri.nz>


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    npm \
    cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get install -y nodejs

RUN pip install numpy 

# Setup user for jupyter
ARG NB_USER="pumahu"
ARG NB_UID="1000"
ENV NB_USER=$NB_USER NB_UID=$NB_UID HOME=/home/$NB_USER
ENV PATH="$HOME/.local/bin:${PATH}"
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER 

RUN mkdir -p /opt/data && \
    chown -R $NB_USER /opt/data

USER $NB_USER
RUN mkdir -p $HOME/pumahu
WORKDIR $HOME/pumahu

COPY --chown=$NB_USER:users ./requirements.txt .
RUN pip install -r requirements.txt --user

COPY --chown=$NB_USER:users . .
RUN python setup.py develop --user 

WORKDIR $HOME

RUN jupyter labextension install jupyterlab-plotly@4.14.3 && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3

CMD jupyter lab --no-browser --ip=* --notebook-dir=$HOME/pumahu/ --config=$HOME/pumahu/src/pumahu/notebook/jupyter_notebook_config.py
