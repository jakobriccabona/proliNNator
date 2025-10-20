# proliNNator
a graph convolutional network based proline prediction tool for protein structures
<img width="612" alt="graph-generation" src="https://github.com/user-attachments/assets/43cddf8a-1fb6-4611-bbbd-bc0b6980dd78" />
```
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input PDB file
  -m MODEL, --model MODEL
                        Path to the model (default: 3D-model-v2.5.keras)
  -p OUTPUT, --pdb OUTPUT
                        Name of the output PDB file (default: output.pdb)
  --csv CSV              
                        Filename to save a csv file with the probabilities (default: output.csv)
  --ramachandran RAMACHANDRAN  
                        Filename to save a Ramachandran plot with probabilities as a PNG (default: ramachandran.png)
  --fastrelax           Flag to perform a fast relax on the structure before analysis
```

this network was trained on fastrelaxed proteins from the CATH database (maximum length 150 amino acids)
## network architecture
<img width="516" alt="architecture" src="https://github.com/user-attachments/assets/b0d28c58-543c-4532-a905-46f6244da02f" />


## docker execution
here is how you can build the docker image:
```
docker build -t prolinnator:v1 .
```
the following command executes a test run:
```
docker run --rm -v $(pwd):/proliNNator/data prolinnator:v1 python proliNNator.py \
-i test/3ft7.pdb -p test/out.pdb -m 3D-model-v2.5.keras --csv test/out.csv\
--ramachandran test/plot.png
```