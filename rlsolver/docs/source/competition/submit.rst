Submit
======

This section explains how to submit your solution. It includes three parts:

1. Neural Networks  
2. Code Packaging  
3. Solution Format

1. Neural Networks
------------------

If your method involves training (e.g., GNN or RL agent), please include:

- `model.py`: model definition  
- `train.py`: training script  
- `inference.py`: inference script  
- `checkpoints/`: folder with model weights (e.g., `best_model.pth`)  

Example command:

.. code-block:: bash

   python inference.py --checkpoint checkpoints/best_model.pth --input data/gset_14.txt --output result/result.txt

2. Code Packaging
------------------

Please include:

- Python source files (`.py`)  
- `requirements.txt` with all dependencies  
- A brief `README.md` with install and run instructions


3. Solution Format
------------------

Your output should be saved in a file called `result.txt` inside the `result/` folder.

Each line represents the assignment of a node to one of two sets (for MaxCut):

.. code-block:: text

   1 2
   2 1
   3 2
   4 1
   5 2

- The first number is the node ID (starting from 1)  
- The second number is the assigned set (1 or 2)  

This format directly follows the example in the README. Make sure to include all nodes and follow the naming exactly.