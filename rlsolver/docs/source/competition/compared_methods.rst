Compared Methods
===============

This section lists all baseline and reference methods that will be compared against your submission. A brief description, advantages, and disadvantages are provided for each.

1. **Classical Heuristics**  
   - **Greedy Heuristic**  
     - **Description**: Sort edges by weight in descending order. Iteratively assign nodes to one of the two partitions if doing so increases the cut.  
     - **Advantages**: Very fast; trivial to implement.  
     - **Disadvantages**: Easily trapped in local optima; quality is often far from optimal on dense graphs.
   - **Local Search (LS)**  
     - **Description**: Start from a random initial cut. At each step, move a single node if it improves the cut value; repeat until no improvement is possible.  
     - **Advantages**: Typically outperforms pure greedy on small to medium graphs.  
     - **Disadvantages**: Performance heavily depends on the initial cut; may require many iterations.

2. **Metaheuristic Methods**  
   - **Simulated Annealing (SA)**  
     - **Description**: Similar to local search but accepts worse moves with a probability that decreases over time (temperature parameter).  
     - **Advantages**: Can escape local optima by allowing uphill moves early on.  
     - **Disadvantages**: Requires careful temperature scheduling; runtime can be large for good solutions.
   - **Genetic Algorithm (GA)**  
     - **Description**: Maintain a population of cuts, apply crossover and mutation operators, and select the best individuals over generations.  
     - **Advantages**: Explores a diverse set of solutions; good for global search.  
     - **Disadvantages**: Many hyperparameters (population size, crossover rate, mutation rate); can be slow to converge.

3. **Exact or MIP Solvers**  
   - **Gurobi (ILP Formulation)**  
     - **Description**: Formulate MaxCut as a binary quadratic integer program (QIP) or an ILP via linearization, solve with Gurobi to optimality or time limit.  
     - **Advantages**: Provides an optimality guarantee when it converges; widely used in industry.  
     - **Disadvantages**: Not scalable to large graphs (≥ 500 nodes) under reasonable time limits.
   - **CPLEX (ILP or QIP)**  
     - **Description**: Similar to Gurobi but using IBM’s CPLEX solver.  
     - **Advantages**: Comparable performance to Gurobi in many instances.  
     - **Disadvantages**: Same scalability constraints on large graphs.

4. **Graph Neural Network (GNN) Methods**  
   - **GCN + MIP Refinement**  
     - **Description**: Use a Graph Convolutional Network (GCN) to predict soft node assignments, then solve a small mixed-integer program to finalize the cut.  
     - **Advantages**: Leverages learned patterns; often yields good solutions on moderate-sized graphs.  
     - **Disadvantages**: Two-stage pipeline adds complexity; training can be time-consuming.
   - **Graph Attention Network (GAT) + Local Search**  
     - **Description**: Use GAT to produce initial node embedding, then apply local search to improve the cut.  
     - **Advantages**: Attention mechanism can capture important graph structure; local search refines results.  
     - **Disadvantages**: Requires tuning of both GAT and local search hyperparameters.

5. **Reinforcement Learning Methods**  
   - **DQN (Deep Q-Network)**  
     - **Description**: Model MaxCut as a sequential decision process. State = partial cut assignment, action = flip node partition, reward = increase in cut value.  
     - **Advantages**: Learns to make decisions that maximize long-term cut value; no human-crafted heuristics needed.  
     - **Disadvantages**: Training can be unstable; high variance; expensive on large graphs.
   - **GNN-PPO (Graph Neural Network + Proximal Policy Optimization)**  
     - **Description**: Use a GNN to encode the graph state and a PPO agent to decide node flips or assignments.  
     - **Advantages**: PPO is more stable than vanilla policy gradient; GNN captures graph structure.  
     - **Disadvantages**: Requires a large number of episodes to converge; sensitive to hyperparameters.
   - **DeepGL (Deep Graph Learning)**  
     - **Description**: A hybrid of GNN and classical local search. Pretrain a GNN to estimate node contributions, then use those estimates to guide a local search routine.  
     - **Advantages**: Balances learned heuristics with search-based refinement; often faster convergence.  
     - **Disadvantages**: Complexity increases due to mixing two paradigms; tuning required for both components.

6. **Comparison Table**  
  
+------------------------+-------------------------------+---------------------------------------------+
| Method                 | Advantages                    | Disadvantages                               |
+========================+===============================+=============================================+
| Greedy Heuristic       | Fast, easy to implement       | Prone to local optima                       |
+------------------------+-------------------------------+---------------------------------------------+
| Local Search (LS)      | Better quality than greedy    | Sensitive to initial solution               |
+------------------------+-------------------------------+---------------------------------------------+
| Simulated Annealing    | Can escape local optima       | Requires temperature tuning                 |
+------------------------+-------------------------------+---------------------------------------------+
| Gurobi                 | Optimality guarantee          | Not scalable to large graphs                |
+------------------------+-------------------------------+---------------------------------------------+
| GCN + MIP Refinement   | Learns graph patterns         | Two-stage pipeline complexity               |
+------------------------+-------------------------------+---------------------------------------------+
| DQN                    | No hand-crafted heuristics    | Unstable training, high variance            |
+------------------------+-------------------------------+---------------------------------------------+
| GNN-PPO                | Stable policy updates (PPO)   | Requires many training episodes             |
+------------------------+-------------------------------+---------------------------------------------+
| DeepGL                 | Combines learning + search    | Complex to tune; mixed methodology overhead |
+------------------------+-------------------------------+---------------------------------------------+