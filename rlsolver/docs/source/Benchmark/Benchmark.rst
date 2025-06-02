Benchmark
=========

This section presents the evaluation results of graph MaxCut algorithms under two settings:  
(1) **distribution-wise**, using the BA distribution;  
(2) **instance-wise**, using the Gset dataset.

1. Distribution-wise Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div style="overflow-x: auto;">

.. csv-table:: Table 1: Results for graph MaxCut on BA distribution
   :header: Nodes, Gurobi, S2V-DQN, S2V-DQN#Gurobi, ECO-DQN, ECO-DQN#Gurobi, Ours, Ours#Gurobi
   :widths: 6, 8, 8, 10, 8, 10, 8, 10

   100, 283.7, 272.3, -4.0%, 283.63, -0.02%, 283.7, 0
   200, 583.3, 557.2, -4.5%, 581.7, -0.26%, 582.2, -0.17%
   300, 880.4, 825.4, -6.2%, 873.9, -0.74%, 878.0, -0.27%
   400, 1179.2, 1100.6, -6.7%, 1165.3, -1.18%, 1174.2, -0.42%
   500, 1477.6, 1374.9, -6.9%, 1167.2, -1.0%, 1471.4, -0.42%
   600, 1774.5, 1647.2, -7.1%, 1752.6, -1.2%, 1769.1, -0.3%
   700, 2068.6, 1907.3, -7.8%, 2043.3, -1.2%, 2062.6, -0.3%
   800, 2361.0, 2182.1, -7.6%, 2331.3, -1.3%, 2358.7, -0.1%
   900, 2655.9, 2425.5, -8.7%, 2616.9, -1.5%, 2647.0, -0.3%
   1000, 2952.2, 2706.1, -8.3%, 2911.9, -1.3%, 2940.3, -0.4%

.. raw:: html

    </div>

.. note::

   The relative difference (columns with "#Gurobi") represents the performance gap with Gurobi's result. Higher values are better.

2. Instance-wise Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div style="overflow-x: auto;">

.. csv-table:: Table 5: Results for graph MaxCut on the Gset dataset in instance-wise scenario
   :header: Graph, Nodes, Edges, BLS, DSDP, KHLWG, RUN-CSP, PI-GNN, iSCO, dREINFORCE, MCPG, Jumanji
   :widths: 6, 6, 8, 8, 8, 8, 8, 8, 8, 10, 8, 8
   :stub-columns: 1

   G14, 800, 4694, 3064, -, 2922, 3061, 2943, 3056, 3064, 3064, 3064
   G15, 800, 4661, 3050, 2938, 3050, 2928, 2990, 3046, 3050, 3050, 2979
   G22, 2000, 19990, 13359, 12960, 13359, 13028, 13181, 13289, 13359, 13359, 13261
   G49, 3000, 6000, 6000, 6000, 6000, 6000, 5918, 5940, 6000, 6000, 5987
   G50, 3000, 6000, 5880, 5880, 5880, 5880, 5820, 5874, 5880, 5880, 5872
   G55, 5000, 12468, 10294, 9960, 10236, 10116, 10138, 10218, 10298, 10296, 10283
   G70, 10000, 9999, 9541, 9456, 9458, -, 9421, 9442, 9586, 9578, 9554

.. raw:: html

    </div>

.. note::

   PI-GNN and Jumanji follow Pattern I; iSCO, dREINFORCE, and MCPG follow Pattern II.
