<<<<<<< HEAD
Graph Datasets
==============

The following table lists common graph datasets and the combinatorial optimization problems (CO problems) they support:

.. list-table::
   :header-rows: 1
   :widths:  15 85

   * - **Dataset**
     - **Supported CO Problems**
   * - BA
     - Maxcut, Graph Partitioning, Graph Coloring, MIS
   * - ER
     - Maxcut, Graph Partitioning, Graph Coloring, MIS
   * - PL
     - Maxcut, Graph Partitioning, Graph Coloring, MIS
   * - `GSET <https://web.stanford.edu/~yyye/yyye/Gset/>`_
     - Maxcut, Graph Partitioning, Graph Coloring, MIS  
       *(supports multiple graph CO problems)*
   * - TSPLIB
     - Traveling Salesman Problem (TSP)
   * - Solomon Instances
     - Vehicle Routing Problem (VRP)
   * - Google Sycamore Circuits
     - Tensor Network Contraction Optimization (TNCO)
=======
Graph Data
==========

本节介绍项目中使用的图结构数据集。

ER 图（Erdős–Rényi）
--------------------

随机图，每条边以固定概率独立生成。用于测试算法在无结构网络上的表现。

BA 图（Barabási–Albert）
------------------------

具有“富者越富”特性的无尺度网络，常见于社交网络模拟。

TSP 图（旅行商问题图）
----------------------

节点为平面坐标点，边权为欧几里得距离。常用于路径规划优化。

Physics Network
---------------

来自真实物理系统的图结构，例如粒子互联图，用于最大割问题测试。

MemeTracker 社交图
------------------

基于社交网络传播路径构建的大规模图，用于顶点覆盖问题。
>>>>>>> 5ec3b7bf06c4ccbf7e7b00bb7ea0862efa6de18c
