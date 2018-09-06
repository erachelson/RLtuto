FROM jaimeps/rl-gym
MAINTAINER Jonathan Sprauel


ENV DEBIAN_FRONTEND noninteractive

RUN pip install  --upgrade pip
RUN pip install scikit-image
RUN pip install pyarrow
RUN pip install PyOpenGL
RUN pip install JSAnimation
RUN pip install ipywidgets
RUN pip install pyglet==1.2.4

USER root
RUN apt update
RUN apt-get update \
     && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        g++  \
        git  \
        curl  \
        cmake \
        zlib1g-dev \
        libjpeg-dev \
        xvfb \
        libav-tools \
        xorg-dev \
        libboost-all-dev \
        libsdl2-dev \
        swig \
		python-opengl \
        python3  \
        python3-dev  \
        python3-future  \
        python3-pip  \
        python3-setuptools  \
        python3-wheel  \
        python3-tk \
        libopenblas-base  \
        libatlas-dev  \
        cython3  \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*


RUN pip install gym[all]

EXPOSE 8888
# Enable jupyter widgets
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

ENV DEBIAN_FRONTEND teletype

# Jupyter notebook with virtual frame buffer for rendering
CMD cd /ds \
    && xvfb-run -s "-screen 0 1400x900x24" \
    /usr/local/bin/jupyter notebook \
    --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.password='sha1:c71acc05a537:9e8076f593c19e80339951e51c952244c7c01b52'