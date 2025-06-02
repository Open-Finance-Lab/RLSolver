Quickstart
==========

The license is **MIT License**.  
The following processes show how to run the algorithm in four parts.

Read Data
-----------------

There are two types of datasets used:

- **Gset**: Provided by Stanford University, stored in the `data/` folder. The number of nodes ranges from 800 to 10,000.

- **Syn**: Synthetic data. Number of nodes ranges from 100 to 1000 across three distributions: Barabasi–Albert (BA), Erdos–Renyi (ER), and Powerlaw (PL). Each distribution has 10 graph instances.

Example: `gset_14.txt` (an undirected graph with 800 nodes and 4694 edges):

.. code-block:: text

   800 4694          # #nodes = 800, #edges = 4694
   1 7 1             # node 1 connects with node 7, weight = 1
   1 10 1            # node 1 connects with node 10, weight = 1
   1 12 1            # node 1 connects with node 12, weight = 1


Store Solution
----------------------

The solution will be stored in the folder `result`.  
Take graph maxcut as an example. The result includes the objective value, number of nodes, algorithm name, and the solution.

Example result for `data/BA_100_ID0.txt` stored in `result/BA_100_ID0.txt`:

.. code-block:: text

   // obj: 273.0                   # objective value
   // running_duration: 71.9577648639679
   // num_nodes: 100
   // alg_name: greedy

   1 1   # node 1 in set 1
   2 2   # node 2 in set 2
   3 2   # node 3 in set 2
   4 2   # node 4 in set 2
   5 2   # node 5 in set 2
   ...

Distribution-wise
--------------------------

1. **Select problem**  

In `config.py`, we select a CO problem:

.. code-block:: python

   PROBLEM = Problem.maxcut  # We can select a problem such as maxcut.

2. **Training**  

Take S2V-DQN as an example. In `config.py`, we set parameters as follows:

.. code-block:: python

   ALG = Alg.s2v
   TRAIN_INFERENCE = 0  # 0 = train, 1 = inference
   NUM_TRAIN_NODES = 200

Run method in command line:

.. code-block:: bash

   python methods/eco_s2v/main.py   # train S2V-DQN
   python methods/L2A/main.py       # run dREINFORCE

3. **Inference** 
 
In the inference stage, we should select dataset(s). Take S2V-DQN as an example:

.. code-block:: python

   TRAIN_NODES_IN_INFERENCE = 200
   directory_data = '../data/syn_BA'  # the directory of datasets
   prefixes = ['BA_100_']             # select the BA graphs with 100 nodes

In `config.py`, we set the parameters:

.. code-block:: python

   TRAIN_INFERENCE = 1  # 0 = train, 1 = inference

Run method in command line:

.. code-block:: bash

   python methods/eco_s2v/main.py   # inference S2V-DQN
   python methods/L2A/main.py       # run dREINFORCE

Instance-wise
----------------------

1. **Select problem**  

In `config.py`, we select a CO problem:

.. code-block:: python

   PROBLEM = Problem.maxcut

2. **Select dataset(s)**  

In `config.py`, we select dataset(s):

.. code-block:: python

   directory_data = '../data/syn_BA'  # the directory of datasets
   prefixes = ['BA_100_']             # select the BA graphs with 100 nodes

3. **Run method**  

Run method in command line:

.. code-block:: bash

   python methods/greedy.py                  # run greedy
   python methods/gurobipy.py                # run gurobi
   python methods/simulated_annealing.py     # run simulated annealing
   python methods/mcpg.py                    # run MCPG
   python methods/iSCO/main.py               # run iSCO
