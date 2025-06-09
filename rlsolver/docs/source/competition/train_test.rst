Train–Test
==========

RLSolver does not require separate training scripts for conventional methods. You can directly run provided solvers on predefined datasets.

Please refer to the `README.md` for details on:

1. Dataset Descriptions
------------------------

- Gset dataset (real-world graphs from Stanford)
- Synthetic datasets (BA, ER, PL distributions)

Each dataset is stored as `.txt` files inside the `data/` folder. The format of each file is:

.. code-block:: text

   800 4694         # number of nodes and edges
   1 7 1            # node 1 connects with node 7 with weight 1
   1 10 1
   ...

2. Running Algorithms
---------------------

Example: Run a solver on a selected dataset

.. code-block:: bash

   python methods/greedy.py
   python methods/L2A/main.py

Before running, make sure the dataset directory and file prefixes are correctly set in the script. For example:

.. code-block:: python

   directory_data = '../data/syn_BA'
   prefixes = ['BA_100_']

3. Output Format
----------------

The final result should be written to `result/result.txt`. Each line represents a node and its assigned label:

.. code-block:: text

   1 2
   2 1
   3 2
   ...

This indicates the partitioning of nodes for the MaxCut solution.

4. Evaluation
-------------

See the `Evaluation` section of the documentation for information on how solutions are assessed.