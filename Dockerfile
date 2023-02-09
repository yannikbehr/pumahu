FROM python:3.7 as builder

LABEL maintainer="Yannik Behr <y.behr@gns.cri.nz>"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    gfortran \
    cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

# update PATH environment variable just to stop the warnings
#ENV PATH=/root/.local/bin:$PATH

# Setup new user
ARG NB_USER="pumahu"
ARG NB_UID="1000"
ENV NB_USER=$NB_USER NB_UID=$NB_UID HOME=/home/$NB_USER
ENV PATH="$HOME/.local/bin:${PATH}"
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER 

RUN mkdir -p /opt/data && \
    chown -R $NB_USER /opt/data

USER $NB_USER
WORKDIR $HOME

COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --user numpy \
    && pip install --user -r requirements.txt \
    && rm -rf  .cache/pip

RUN mkdir -p $HOME/pumahu
WORKDIR $HOME/pumahu

COPY --chown=$NB_USER:users . .
RUN python setup.py develop --user 

WORKDIR $HOME
EXPOSE 8061
EXPOSE 8050
CMD ["heat_uks", "--rdir", "/opt/data", "-s", "2016-03-04", "-f", "-p", "-d"]
