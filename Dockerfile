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

#Download the models & python file for execution from my github repo
RUN git clone https://github.com/jakobriccabona/proliNNator.git

WORKDIR /proliNNator

#Download dependencies
RUN pip install --no-cache-dir numpy pandas argparse tensorflow==2.12.0 \
 scikit-learn matplotlib seaborn spektral \
 pyrosetta-installer \
 menten-gcn
RUN python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'

#define the command to run your script
CMD ["python", "proliNNator.py"]
