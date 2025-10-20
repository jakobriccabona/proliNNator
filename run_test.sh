#!/bin/bash

sudo docker run --rm -v $(pwd):/proliNNator/data prolinnator:v1 python proliNNator.py \
-i test/3ft7.pdb -p test/out.pdb -m 3D-model-v2.5.keras --csv test/out.csv\
--ramachandran test/out.png

