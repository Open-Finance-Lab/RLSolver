## Combinatorial optimization with physics-inspired graph neural networks

This repository is an ongoing implementation of [Combinatorial optimization with physics-inspired graph neural networks](https://www.nature.com/articles/s42256-022-00468-6) done with PyTorch Lightning and PyTorch Geometric.

The current version implements only the Maxcut and Maximum Independent Set (MIS) problems on random d-regular graphs, as explained in the paper. Please beware that the results are still inconclusive and this is an ongoing implementation, your feedback is more than welcomed.




### Running the code

The file structure is quite simple and straightfoward. ```main.py``` is the file used to run the experiments, you can find the various arguments controlling the runs inside. ```data.py``` generates a given number of random d-regular graphs, ```models.py``` contains the GNN model with various possible architectures, and ```util.py``` contains the code to compute the Hamiltonians for each problem. As a simple example, you can run the following commands to check that the code runs:

```bash
python main.py --maxcut --epochs 3 # maxcut
python main.py --epochs 3 # maxcut
```