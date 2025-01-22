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
<img width="612" alt="graph-generation" src="https://github.com/user-attachments/assets/43cddf8a-1fb6-4611-bbbd-bc0b6980dd78" />
<img width="516" alt="architecture" src="https://github.com/user-attachments/assets/b0d28c58-543c-4532-a905-46f6244da02f" />
