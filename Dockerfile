# syntax=docker/dockerfile:1

# Use an official NVIDIA runtime as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Create and set the working directory
WORKDIR /

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    unzip \
    git \
    gcc \
    g++ \
    libopenblas-dev \
    python3-pip \
    python3-dev \
    zlib1g \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link from python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Download your repo
RUN git clone https://github.com/jakobriccabona/proliNNator.git

WORKDIR /proliNNator

# Install PyTorch + CUDA 11.8
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install torch_geometric + dependencies
RUN pip install --no-cache-dir pyg-lib==0.3.1 \
    --extra-index-url https://data.pyg.org/whl/cu118

RUN pip install --no-cache-dir torch_geometric

# Install remaining dependencies
RUN pip install --no-cache-dir \
    biopython \
    numpy \
    matplotlib \
    scikit-learn

# Default command
CMD ["python", "proliNNator.py"]