Welcome to RLSolver!
====================

`RLSolver <https://github.com/zhumingpassional/RLSolver>`_ : GPU-based Massively Parallel Environments for Combinatorial Optimization (CO) Problems Using Reinforcement Learning.

We aim to showcase the effectiveness of massively parallel environments for CO problems using RL. With GPU-based parallel environments, sampling speed is significantly improved.

Overview
--------

RLSolver has three layers:

- Environments: providing massively parallel environments using GPUs.
- RL agents: providing RL algorithms, e.g., REINFORCE.
- Problems: typical CO problems, e.g., graph maxcut and TNCO.

Key Technologies
----------------

- **GPU-based massively parallel environments** using CUDA.
- **Distribution-wise** methods (e.g. ECO, S2V) are faster than instance-wise methods (e.g. MCPG, iSCO).

Installation
------------

.. code-block:: bash

   pip install rlsolver --upgrade

Or from GitHub:

.. code-block:: bash

   git clone https://github.com/zhumingpassional/RLSolver
   cd RLSolver
   pip install .

----

.. toctree::
   :maxdepth: 1
   :caption: Overview

   about/overview
   about/cloud
   about/parallel

.. toctree::
   :maxdepth: 2
   :caption: HelloWorld

   helloworld/hello
   helloworld/net
   helloworld/agent
   helloworld/env
   helloworld/run
   helloworld/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Methods

   Methods/index

.. toctree::
   :maxdepth: 2
   :caption: Algorithms

   algorithms/distribution-wise
   algorithms/instance-wise

.. toctree::
   :maxdepth: 2
   :caption: Datasets

   Datasets/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/config
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/ECO-DQN 求解 maxcut

.. toctree::
   :maxdepth: 2
   :caption: RLSolver System

   RLSolver/overview
   RLSolver/helloworld
   RLSolver/datasets
   RLSolver/environments
   RLSolver/benchmarks

----

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
