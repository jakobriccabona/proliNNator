# syntax=docker/dockerfile:1

# use an official nvidia runtime as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

#create and set the working directory
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
 zlib1g \
 vim \
 && rm -rf /var/lib/apt/lists/*

# Create a symbolic link from python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Download your repo
RUN git clone -b new https://github.com/jakobriccabona/proliNNator.git

WORKDIR /proliNNator

# Install remaining dependencies
RUN pip install biopython torch torchvision torchaudio pyg-library torch_geometric numpy matplotlib scikit-learn

# Default command
CMD ["python", "proliNNator.py"]