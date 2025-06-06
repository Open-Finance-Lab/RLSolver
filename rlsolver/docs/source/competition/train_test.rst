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

Distribution-wise
-----------------

1. **Graph distribution and data directory**  
   In `config.py`, set:
   .. code-block:: python

      from rlsolver.methods.config import GraphType

      GRAPH_TYPE = GraphType.BA      # or GraphType.ER, GraphType.PL
      DATA_DIR   = rlsolver_path + "/data/syn_" + GRAPH_TYPE.value

2. **Train vs. Inference**  
   In `config.py`, set one of:
   .. code-block:: python

      TRAIN_INFERENCE = 0    # to train on the entire distribution
      TRAIN_INFERENCE = 1    # to run inference on all prefixes

3. **Prefixes for batch inference**  
   In `config.py`, replace the default prefixes with:
   .. code-block:: python

      NUM_INFERENCE_NODES = 
          [100, 200, 300, 400, 500,
          600, 700, 800, 900, 1000,
          1100, 1200, 2000, 3000,
          4000, 5000, 10000]

      INFERENCE_PREFIXES = 
         [ GRAPH_TYPE.value + "_" + str(n) + "_"
          for n in NUM_INFERENCE_NODES ]

4. **Run the script**  
   .. code-block:: console

      python methods/L2A/main.py

   - If `TRAIN_INFERENCE = 0`, this trains one model over the entire synthetic distribution.  
   - If `TRAIN_INFERENCE = 1`, this iterates over every prefix in `INFERENCE_PREFIXES` and performs inference.

Example: Train S2V with 20-node BA graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To help you understand the full pipeline, here is a detailed example using `s2v` with 20-node BA graphs.

1. Set basic config in `config.py`:

   .. code-block:: python

      ALG = Alg.s2v
      GRAPH_TYPE = GraphType.BA
      NUM_TRAIN_NODES = 20
      TRAIN_INFERENCE = 0

2. Run training:

   .. code-block:: console

      python methods/eco_s2v/main.py

   This will generate a folder:

   .. code-block:: text

      rlsolver/methods/eco_s2v/pretrained_agent/tmp/s2v_BA_20spin_b/

   Inside this folder, multiple `.pth` model snapshots will be saved over time.

   .. image:: /_static/example_s2v_training.png

3. Select the best model from this folder:

   Edit `config.py` to point to that folder:

   .. code-block:: python

      NEURAL_NETWORK_FOLDER = rlsolver_path + "/methods/eco_s2v/pretrained_agent/tmp/s2v_BA_20spin_b"

   Then run:

   .. code-block:: console

      python methods/eco_s2v/train_and_inference/select_best_neural_network.py

   It will produce a file like:

   .. code-block:: text

      s2v_BA_20spin_best_1033.pth

  .. image:: /_static/best.png


4. Rename and move best model:

   Rename the file to:

   .. code-block:: text

      s2v_BA_20spin_best.pth

   And move it to:

   .. code-block:: text

      rlsolver/methods/eco_s2v/pretrained_agent/

  .. image:: /_static/move.png


5. Switch to inference mode:

   In `config.py`, set:

   .. code-block:: python

      TRAIN_INFERENCE = 1
      NUM_TRAINED_NODES_IN_INFERENCE = 20
      NUM_INFERENCE_NODES = [20, 100, 200, 400, 800]  # Any scale ≥ training size

      NEURAL_NETWORK_SAVE_PATH = rlsolver_path + "/methods/eco_s2v/pretrained_agent/s2v_BA_20spin_best.pth"

6. Run inference:

   .. code-block:: console

      python methods/eco_s2v/main.py

   This performs inference across the node sizes in `NUM_INFERENCE_NODES`.

This completes the full training + model selection + inference pipeline for S2V using BA graphs.



Instance-wise
-------------

1. **Data directory and single prefix**  
   In `config.py`, override to target one file only. For example, to test graph “g22” from Gset:
   .. code-block:: python

      DATA_DIR            = rlsolver_path + "/data/Gset"
      INFERENCE_PREFIXES  = ["g22_"]    # only this one instance
      NUM_INFERENCE_NODES = [2000]      # node count for “g22”
      TRAIN_INFERENCE     = 1

2. **Run on that single instance**  
   .. code-block:: console

      python methods/L2A/main.py

   The script will load `g22.txt` from `DATA_DIR` and perform inference for that one prefix.

3. **Conventional methods (single graph)**  
   For other Instance-wise scripts, specify `--data-dir` and `--prefix`. For example:
   .. code-block:: console

      python methods/greedy.py \
        --data-dir ../data/Gset \
        --prefix g22

      python methods/gurobi.py \
        --data-dir ../data/Gset \
        --prefix g22

      python methods/simulated_annealing.py \
        --data-dir ../data/Gset \
        --prefix g22

      python methods/mcpg.py \
        --data-dir ../data/Gset \
        --prefix g22

      python methods/iSCO/main.py \
        --data-dir ../data/Gset \
        --prefix g22

      python methods/PI-GNN/main.py \
        --data-dir ../data/Gset \
        --prefix g22

      python methods/L2A/main.py \
        --data-dir ../data/Gset \
        --prefix g22 \
        --mode instance


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





