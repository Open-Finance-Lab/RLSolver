# Knapsack Problem Solver

A comprehensive Python implementation of multiple algorithms for solving the 0/1 knapsack problem.

## Overview

This project implements 6 different algorithms for solving the classic 0/1 knapsack optimization problem:

- **Brute Force** - Exhaustive search through all possible combinations
- **Greedy (Cost/Weight Ratio)** - Heuristic approach selecting items by best value-to-weight ratio
- **Dynamic Programming** - Optimal solution using memoization
- **Branch and Bound** - Optimized search with pruning
- **FPTAS** - Fully Polynomial-Time Approximation Scheme
- **Simulated Annealing** - Metaheuristic optimization approach



## Input Format

Instance files (`.txt`) contain one problem per line with the format:
```
<instance_id> <number_of_items> <capacity> <weight1> <cost1> <weight2> <cost2> ...
```

Example:
```
9000 4 100 18 114 42 136 88 192 3 223
```

## Output Format

Solution files (`.txt`) contain one solution per line with the format:
```
<instance_id> <number_of_items> <best_cost> <item1_selected> <item2_selected> ...
```

Where `item_selected` is 1 if the item is included in the knapsack, 0 otherwise.

Example:
```
1 1 0 1
```

## Algorithm Details

### Brute Force
- **Time Complexity**: O(2^n)
- **Space Complexity**: O(n)
- **Optimality**: Guaranteed optimal
- **Best for**: Small problems (n ≤ 20)

### Dynamic Programming
- **Time Complexity**: O(nW) where W is capacity
- **Space Complexity**: O(nW)
- **Optimality**: Guaranteed optimal
- **Best for**: Medium problem### Error Handling

The solver includes robust error handling for:
- Invalid input file formats
- Malformed data lines
- File I/O errors
- Invalid algorithm parameters


### Branch and Bound
- **Time Complexity**: O(2^n) worst case, much better in practice
- **Space Complexity**: O(n)
- **Optimality**: Guaranteed optimal
- **Best for**: Medium to large problems where DP is too memory-intensive

### Greedy (Ratio)
- **Time Complexity**: O(n log n)
- **Space Complexity**: O(n)
- **Optimality**: Approximation (not guaranteed optimal)
- **Best for**: Quick approximate solutions

### FPTAS
- **Time Complexity**: O(n^3/ε) where ε is approximation factor
- **Space Complexity**: O(n^2/ε)
- **Optimality**: (1-ε) approximation
- **Best for**: Large problems requiring near-optimal solutions

### Simulated Annealing
- **Time Complexity**: O(iterations × n)
- **Space Complexity**: O(n)
- **Optimality**: Heuristic (no optimality guarantee)
- **Best for**: Very large problems or when other methods are too slow

## Test Data

The `inst/` directory contains test instances of varying sizes:
- `knap_4.txt` - 4 items (for testing)
- `knap_10.txt` - 10 items
- `knap_15.txt` - 15 items
- `knap_20.txt` - 20 items
- `knap_25.txt` - 25 items
- `knap_30.txt` - 30 items
- `knap_32.txt` - 32 items
- `knap_35.txt` - 35 items
- `knap_37.txt` - 37 items
- `knap_40.txt` - 40 items




