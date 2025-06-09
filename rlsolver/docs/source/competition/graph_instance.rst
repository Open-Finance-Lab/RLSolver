Graph Instance
==============

This competition supports two categories of datasets: **Gset** (real-world instances) and **Syn** (synthetic graphs).

1. **Gset Dataset**

- The Gset dataset is originally released by Stanford University.
- Graphs are stored as `.txt` files in the `data/` folder.
- The number of nodes ranges from **800 to 10000**.
- Each line describes an edge in the format:  

where `u` and `v` are node indices (starting from 1), and `w` is the edge weight.

- The first line of the file describes the total number of nodes and edges.  
For example, the beginning of `gset_14.txt` is:

2. **Syn Dataset (Synthetic)**

- Synthetic graphs are generated with 3 random distributions:
- **BA**: Barabási–Albert
- **ER**: Erdős–Rényi
- **PL**: Power Law

- Node counts range from **100 to 1000**, with 10 graph instances per distribution and size.

- Files are also stored in the `data/` directory. Example:

3. **Directory Layout**

4. **Graph File Format**

Each `.txt` file format:
- Line 1: `N E` (number of nodes, number of edges)
- Line 2 onward: `u v w` (node `u` is connected to node `v` with weight `w`)

Node indices start from **1**, and the graphs are undirected.

5. **Data Selection in Code**

In the code, you can specify dataset by setting:
```python
directory_data = '../data/syn_BA'
prefixes = ['BA_100_']
