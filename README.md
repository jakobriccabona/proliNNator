# proliNNator
a NN based proline prediction tool for protein structures

```
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input PDB file
  -m MODEL, --model MODEL
                        Path to the model
  -o OUTPUT, --output OUTPUT
                        Name of the output PDB file (default: output.pdb)
  --csv CSV              Filename to save a csv file with the probabilities
  --ramachandran RAMACHANDRAN
                        Filename to save a Ramachandran plot with probabilities as a PNG
  --fastrelax           Flag to perform a fast relax on the structure before analysis
```
