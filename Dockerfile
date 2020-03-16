FROM nvidia/cuda:10.1-cudnn7-devel

ADD . /

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	libpng-dev libjpeg-dev python3-opencv ca-certificates \
	python3-dev build-essential pkg-config git curl wget automake libtool sudo && \
  rm -rf /var/lib/apt/lists/*

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
RUN pip install --user numpy pandas tqdm PyYAML \
	'git+https://github.com/rwightman/pytorch-image-models'
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user torch torchvision cython \
	'git+https://github.com/facebookresearch/fvcore'
RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /

# Execute
CMD ["sh", "run.sh"]
