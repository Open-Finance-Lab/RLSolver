Conventional Methods
====================

<<<<<<< HEAD
Below is a brief description of each classical method implemented in RLSolver.

Gurobi
------
Gurobi Optimizer is a highly optimized commercial solver for linear, integer, and quadratic programs.  
For the MaxCut problem, RLSolver uses the Quadratic Unconstrained Binary Optimization (QUBO) formulation rather than the Integer Linear Programming (ILP) formulation, because QUBO typically yields higher solution quality and faster convergence; hence, Gurobi is set to solve the QUBO by default.  
It combines branch-and-bound, cutting planes, presolve reductions, and primal heuristics to efficiently navigate the solution tree and converge on an optimal (or provably near-optimal) solution.

Greedy
------
Greedy Heuristic, A fast, myopic heuristic that builds a solution one step at a time by always selecting the locally best choice.  
Although it lacks global optimality guarantees, its simplicity makes it a strong baseline and a useful warm-start for other methods.

SDP
-----------------------------------------
Semidefinite Programming (SDP) approach lifts the original combinatorial problem into a higher-dimensional matrix space, turning it into a convex SDP.  
Solving the SDP yields a bound on the optimum; randomized rounding of the matrix solution then produces a high-quality feasible solution to the original problem.

SA
------------------------
Inspired by the physical process of slow cooling in metallurgy, Simulated Annealing (SA) explores the solution space by occasionally accepting worse moves.  
The probability of accepting uphill (worsening) moves decreases over time (“temperature” schedule), allowing escape from local minima and gradual convergence.

GA
----------------------
Genetic Algorithm (GA) maintains a population of candidate solutions (chromosomes).  
Each generation applies selection (keeping the fittest), crossover (recombining parts of two parents), and mutation (random small changes) to evolve toward better solutions over many iterations.
=======
本节介绍项目中使用的传统组合优化方法。

Gurobi
------

Gurobi 是一款强大的商业线性规划与整数规划求解器，被广泛应用于优化问题中。我们使用 Gurobi 作为基准方法之一评估强化学习算法的性能。

Greedy
------

贪心算法是一种构造式启发式算法，在每一步选择当前最优解。我们实现了一种基于边权重的贪心策略用于近似求解。

SDP
---

半正定规划（SDP）方法用于求解 MaxCut 等问题，通过松弛组合优化问题到连续空间。

SA
--

模拟退火（Simulated Annealing）是一种基于概率的启发式搜索方法，模仿物理退火过程。

GA
--

遗传算法（Genetic Algorithm）模仿自然选择过程，基于选择、交叉和变异机制寻找最优解。
>>>>>>> 5ec3b7bf06c4ccbf7e7b00bb7ea0862efa6de18c
