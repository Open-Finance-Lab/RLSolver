Evaluate
========

This section describes how different categories of algorithms are evaluated in RLSolver Competition 2025.

**Evaluation Categories**

We evaluate submitted methods according to their type:

1. **Distribution-wise Reinforcement Learning Methods**
   - These are RL-based methods that train on distributions of graph instances and are capable of generalizing to unseen instances.
   - Evaluation metrics include:
     - **Training Time**: Total time required to train the agent across training instances.
     - **Inference Time**: Average time to infer a solution for a single test instance.
     - **Objective Value**: The primary optimization target, e.g., MaxCut value, tour length, etc.

2. **Conventional Methods**
   - These include classical heuristics, local search, greedy algorithms, and solvers like Gurobi or CPLEX.
   - Evaluation metrics include:
     - **Running Time**: Total time taken to solve all test instances (no training phase).
     - **Objective Value**: The final score or cost value obtained by the algorithm.

**Evaluation Criteria Summary**


+-----------------------------+-------------------------+-------------------------+
| Method Type                 | Time Metric             | Optimization Metric     |
+=============================+=========================+=========================+
| Distribution-wise RL        | Training + Inference    | Objective Value         |
+-----------------------------+-------------------------+-------------------------+
| Conventional (non-RL)       | Running Time only       | Objective Value         |
+-----------------------------+-------------------------+-------------------------+

**Notes**

- **Objective Value** is defined by the problem instance (e.g., MaxCut, Set Cover, Knapsack). Higher is better unless otherwise specified.
- **Inference Time** is measured on GPU (if applicable), averaged across all test instances.
- All methods will be evaluated on a standardized server with fixed computational limits.
- Final rankings may use a weighted combination of time and objective metrics, depending on the track.