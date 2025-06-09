Training
========

RLSolver does not require separate training scripts for conventional methods. You can directly run provided solvers on predefined datasets.

Please refer to the `README.md` for additional dataset details.

We demonstrate the full pipeline using the `eco` method on 20-node BA graphs.

The training phase helps the reinforcement learning agent (e.g., `eco`) learn the relationship between graph **topology** and **optimal or near-optimal solutions** (e.g., maximum cut).  
By observing graph structures and receiving rewards based on solution quality, the agent gradually learns a strategy that generalizes to similar unseen graphs.

1. **Set basic config in ``config.py``**:

   .. code-block:: python

      ALG = Alg.eco                                   # select eco as the RL method
      GRAPH_TYPE = GraphType.BA                       # use BA (Barabási–Albert) graph distribution
      NUM_TRAIN_NODES = 20                            # each training graph has 20 nodes
      TRAIN_INFERENCE = 0                             # 0 = train mode; 1 = inference mode

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

      NEURAL_NETWORK_SUBFOLDER = "s2v_BA_20spin_s"

   To select a different model folder, simply replace the string in ``NEURAL_NETWORK_SUBFOLDER`` with the name of the desired folder.  
   For example:

   .. code-block:: python

      NEURAL_NETWORK_SUBFOLDER = "eco_BA_20spin_p"

   Then run:

   .. code-block:: console

      python methods/eco_s2v/train_and_inference/select_best_neural_network.py

   It will generate a file like:

   .. code-block:: text

      eco_BA_20spin_23_best.pth

   .. image:: /_static/best.png

4. **Rename and move the best model**:

   .. code-block:: text

      eco_BA_20spin_best.pth  →  rlsolver/methods/eco_s2v/pretrained_agent/

   .. image:: /_static/move.png

