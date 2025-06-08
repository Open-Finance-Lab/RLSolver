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

This section shows only the configuration fields you must modify for each mode.

Distribution-wise (eco Example)
-------------------------------

We demonstrate the full pipeline using the `eco` method on 20-node BA graphs.

1. **Set basic config in ``config.py``**:

   .. code-block:: python

      ALG = Alg.eco                                   # select eco as the RL method
      GRAPH_TYPE = GraphType.BA          # use BA (Barabási–Albert) graph distribution
      NUM_TRAIN_NODES = 20                # each training graph has 20 nodes
      TRAIN_INFERENCE = 0                     # 0 = train mode; 1 = inference mode


2. **Run training**:

   .. code-block:: console

      python methods/eco_s2v/main.py

   This will generate a folder:

   .. code-block:: text

      rlsolver/methods/eco_s2v/pretrained_agent/tmp/eco_BA_20spin_p/

   Inside this folder, multiple `.pth` model snapshots will be saved over time.

   .. image:: /_static/example_eco_training.png

3. **Select the best model from this folder**:

   Edit ``methods/eco_s2v/config.py``.  
   Find the line:

   .. code-block:: python

      NEURAL_NETWORK_FOLDER = rlsolver_path + "/methods/eco_s2v/pretrained_agent/tmp/" + NEURAL_NETWORK_SUBFOLDER

   Replace only the last part (i.e., the `NEURAL_NETWORK_SUBFOLDER`) with the desired folder name directly.  
   For example, if you want to select the folder named ``eco_BA_20spin_p``, change it to:

   .. code-block:: python

      NEURAL_NETWORK_FOLDER = rlsolver_path + "/methods/eco_s2v/pretrained_agent/tmp/" + "eco_BA_20spin_p"

   Then run:

   .. code-block:: console

      python methods/eco_s2v/train_and_inference/select_best_neural_network.py

   It will generate a file like:

   .. code-block:: text

      eco_BA_20spin_23_best.pth

   .. image:: /_static/best.png


4. **Rename and move best model**:

   .. code-block:: text

      eco_BA_20spin_best.pth  →  rlsolver/methods/eco_s2v/pretrained_agent/

   .. image:: /_static/move.png

5. **Switch to inference mode**:

   In ``config.py``, set:

   .. code-block:: python

      TRAIN_INFERENCE = 1                                              # 1 = inference mode
      NUM_TRAINED_NODES_IN_INFERENCE = 20             # model was trained on 20-node graphs
      NUM_INFERENCE_NODES = [20, 100, 200, 400, 800]   # test on graphs of various sizes

Here, although the model was trained only on 20-node graphs, it can be applied to larger graphs (e.g., 100–800 nodes).
You only need to make sure that all graphs used for inference have node counts greater than or equal to 20.

6. **Run inference**:

Run the following command:

.. code-block:: console

   python methods/eco_s2v/main.py

This step uses the newly selected best neural network parameters to run inference over all test graph instances.

The result files will be saved in:

.. code-block:: text

   rlsolver/result/syn_BA/

Each result file corresponds to one test graph and includes:

- ``obj``: the best objective value (i.e., maximum cut size),
- ``running_duration``: time taken to solve the instance (in **seconds**),
- ``num_nodes``: the number of nodes in the graph,
- ``alg_name``: the algorithm used (e.g., ``eco``),
- followed by the node assignments (each node assigned to group 1 or 2).

Example output:

.. image:: /_static/result.png
   :align: center
   :width: 600px

This completes the full pipeline: **training → model selection → inference** for ``eco`` on distribution-wise BA graphs.

Instance-wise (Greedy Baseline on Gset)
-------------
1. **Set problem and dataset**  

   In ``methods/config.py``, set the following:

   .. code-block:: python

      PROBLEM = Problem.maxcut
      DIRECTORY_DATA = "../data/gset"
      PREFIXES = ["gset_22"]

   This will run the greedy algorithm on the Gset instance ``gset_22.txt``.

2. **Run Greedy**  

   Use the following command to execute the baseline algorithm:

   .. code-block:: console

      python methods/greedy.py

   This script runs `greedy_maxcut()` using the specified file(s) under `data/gset/`.

3. **Result Output**  

After running the greedy algorithm, the results will be saved to:
.. code-block:: text

rlsolver/result/syn_BA/

Each result file corresponds to one test instance and contains:

 Example output:

.. image:: /_static/result2.png
   :align: center
   :width: 600px

- ``obj``: Final objective value (e.g., total cut size for MaxCut).
- ``running_duration``: Time taken by the algorithm in **seconds**.
- ``num_nodes``: Number of nodes in the graph instance.
- ``alg_name``: The algorithm used to generate the solution (e.g., greedy).
- Each following line: node ID and its assigned label (partition).  
  For MaxCut, labels represent two sets in the cut.

The file is automatically generated and named based on the instance prefix and a unique suffix, such as:

.. code-block:: text

   BA_100_ID0_3.txt

You can find all greedy results in the ``result/syn_BA`` folder.