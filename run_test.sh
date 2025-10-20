#!/bin/bash

sudo docker run --rm -v $(pwd):/proliNNator/data prolinnator:v1 python proliNNator.py \
-i data/3ft7.pdb -p data/out.pdb -m 3D-model-v2.5.keras --csv data/out.csv\
--ramachandran data/out.png

