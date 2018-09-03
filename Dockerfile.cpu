FROM conda/miniconda3

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN conda install -y numpy pyyaml mkl setuptools cmake cffi
RUN conda install -y pytorch-cpu torchvision-cpu -c pytorch

RUN apt-get update -y && \
		apt-get install -y --no-install-recommends \
			git \
      wget \
      build-essential \
		&& \
		apt-get clean && \
		rm -rf /var/lib/apt/lists/*


RUN conda install -y jupyter matplotlib scikit-learn nltk bokeh scikit-image

WORKDIR /work
CMD ["bash"]
