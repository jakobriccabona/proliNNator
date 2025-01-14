# proliNNator
a graph convolutional network based proline prediction tool for protein structures

```
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input PDB file
  -m MODEL, --model MODEL
                        Path to the model (default: 3D-model-v2.4.keras)
  -o OUTPUT, --output OUTPUT
                        Name of the output PDB file (default: output.pdb)
  --csv CSV              Filename to save a csv file with the probabilities
  --ramachandran RAMACHANDRAN
                        Filename to save a Ramachandran plot with probabilities as a PNG
  --fastrelax           Flag to perform a fast relax on the structure before analysis
```

this network was trained on fastrelaxed proteins (maximum length 150 amino acids)
