RL Methods
==========

本节介绍我们使用的强化学习方法。

ECO
---

ECO（Embedding-based Combinatorial Optimization）方法结合图嵌入和策略网络，用于求解结构化优化问题。

S2V
---

S2V（Structure2Vec + Q-learning）利用图嵌入神经网络和强化学习训练贪心策略。

MCPG
----

Monte Carlo Policy Gradient 方法使用策略梯度和采样方法优化图上的节点选择问题。

ISCO
----

Improved S2V with Curriculum Optimization，结合任务难度控制与图嵌入学习以提高泛化能力。

Jumanji
-------

基于 JAX 的强化学习优化平台，支持组合优化任务的可微分求解与训练。
