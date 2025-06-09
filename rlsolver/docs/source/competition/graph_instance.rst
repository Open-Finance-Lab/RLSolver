Graph Instance
==============

Synthetic graphs are generated with three random distributions:

- **BA**: Barabási–Albert
- **ER**: Erdős–Rényi
- **PL**: Power Law

- Node counts range from **100 to 1000**, with 10 graph instances per distribution and size.
- Files are stored in the `data/` directory. Examples include:

  - `BA_100_ID0.txt`
  - `ER_100_ID0.txt`
  - `PL_100_ID0.txt`

**File Format (Example: ``BA_100_ID0.txt``)**

Each graph file is a plain text file with the following format:

.. code-block:: text

   100 384
   1 2 1
   1 3 1
   1 4 1
   1 5 1
   1 6 1
   ...
   1 40 1

Explanation:

- The **first line** contains two integers:
  - `100`: the number of nodes in the graph.
  - `384`: the total number of edges.

- Each **subsequent line** represents an undirected edge with a unit weight:
  - The first number is the **source node index**.
  - The second number is the **target node index**.
  - The third number is the **edge weight** (always 1 in synthetic datasets).

- **Node indices are 1-based**, meaning nodes are labeled from `1` to `100`.

- An edge `1 2 1` means there is an edge between node 1 and node 2 with weight 1.  
  Since the graph is undirected, the edge `2 1 1` is not repeated.

This format is consistent across all synthetic graphs (BA, ER, PL) in the dataset.
