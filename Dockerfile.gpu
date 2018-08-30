FROM nvidia/cuda:8.0-devel-ubuntu16.04

ARG CONDA_DIR=/opt/conda
ARG CONDA_DOWNLOAD_SCRIPT_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update -y && \
		apt-get install -y --no-install-recommends \
			git \
      wget \
      ca-certificates \
      build-essential \
		&& \
		apt-get clean && \
		rm -rf /var/lib/apt/lists/*

RUN wget $CONDA_DOWNLOAD_SCRIPT_URL -qO /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*

RUN export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN conda install -y numpy pyyaml mkl setuptools cmake cffi
RUN conda install -y -c pytorch magma-cuda80

RUN git clone --recursive https://github.com/pytorch/pytorch /tmp/pytorch
RUN cd /tmp/pytorch/ && python3 setup.py install

RUN pip install https://github.com/pytorch/text/archive/master.zip

RUN conda install -y torchvision -c pytorch
RUN conda install -y jupyter matplotlib scikit-learn nltk bokeh scikit-image

WORKDIR /work
CMD ["bash"]
