# Usage:
# To build without AWS-EFA:
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_BUILD_MODE=no-efa .
# To build with AWS-EFA:
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_BUILD_MODE=efa .

ARG COSMOS_RL_BUILD_MODE=efa

ARG CUDA_VERSION=12.8.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS no-efa-base

ARG GDRCOPY_VERSION=v2.4.4
ARG EFA_INSTALLER_VERSION=1.42.0
ARG AWS_OFI_NCCL_VERSION=v1.16.0
# NCCL version, should be found at https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/
ARG NCCL_VERSION=2.26.2-1+cuda12.8
ARG PYTHON_VERSION=3.12

ENV TZ=Etc/UTC

RUN apt-get update -y && apt-get upgrade -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    curl git gpg lsb-release tzdata wget
RUN apt-get purge -y cuda-compat-*
RUN apt-get update && apt-get install -y dnsutils

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:$LIBRARY_PATH
ENV PATH=/opt/gdrcopy/bin:$PATH

###################################################
## Install NCCL with specific version
RUN apt-get remove -y --purge --allow-change-held-packages \
    libnccl2 \
    libnccl-dev
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update -y \
    && apt-get install -y libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION}

###################################################
## Install cuDNN
RUN apt-get update -y && \
    apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

###################################################
## Install redis
# Download and add Redis GPG key, Redis APT repository
RUN curl -fsSL https://packages.redis.io/gpg  | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb  $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list

# Update package list
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y redis-server

###################################################
RUN apt-get install -qq -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
## Install python
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y --allow-change-held-packages \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
## Create a virtual environment

RUN python${PYTHON_VERSION} -m venv /opt/venv/cosmos_rl
ENV PATH="/opt/venv/cosmos_rl/bin:$PATH"

RUN pip install -U pip setuptools wheel packaging

# even though we don't depend on torchaudio, vllm does. in order to
# make sure the cuda version matches, we install it here.
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /workspace/cosmos_rl/requirements.txt

# Install flash_attn separately
# RUN pip install flash_attn==2.8.2 --no-build-isolation


RUN pip install \
    torchao==0.13.0 \
    flash_attn==2.8.3 \
    -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly \
    flashinfer-python \
    transformer_engine[pytorch] \
    -r /workspace/cosmos_rl/requirements.txt

# TODO: (lms) remove nightly version of vllm and triton in later vllm release.
# Here we install nightly version of triton in pytorch nightly index.
# and install triton_kernels from vllm gpt-oss index, because vllm gpt-oss needs 
# some triton kernels. Install triton and triton_kernels after vllm installation
# to avoid version error.

# Install triton and triton_kernels
RUN pip uninstall -y triton triton_kernels && \
    pip install -U triton --pre --extra-index-url https://download.pytorch.org/whl/nightly --no-deps && \
    pip install -U triton_kernels --extra-index-url https://wheels.vllm.ai/gpt-oss/ --no-deps
    
###################################################
FROM no-efa-base AS efa-base

# Remove HPCX and MPI to avoid conflicts with AWS-EFA
RUN rm -rf /opt/hpcx \
    && rm -rf /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

RUN apt-get remove -y --purge --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1

###################################################
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

###################################################
## Install AWS-OFI-NCCL plugin
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libhwloc-dev
#Switch from sh to bash to allow parameter expansion
SHELL ["/bin/bash", "-c"]
RUN curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
        --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws \
    && make -j $(nproc) \
    && make install \
    && cd .. \
    && rm -rf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && rm aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH


###################################################
## Image target: cosmos_rl
FROM ${COSMOS_RL_BUILD_MODE}-base AS package

COPY . /workspace/cosmos_rl
RUN pip install /workspace/cosmos_rl && rm -rf /workspace/cosmos_rl
