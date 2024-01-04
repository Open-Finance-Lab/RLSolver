import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from util import obj_maxcut
from util import read_nxgraph
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=400, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()




# approx ratio 0.87
def goemans_williamson(graph: nx.Graph):
    n = graph.number_of_nodes() # num of nodes
    edges = graph.edges

    x = cp.Variable((n,n), symmetric = True) #construct n x n matrix

    # diagonals must be 1 (unit) and eigenvalues must be postivie
    # semidefinite
    constraints = [x >> 0] + [ x[i,i] == 1 for i in range(n) ]

    #this is function defing the cost of the cut. You want to maximize this function
    #to get heaviest cut
    objective = sum( (0.5)* (1 - x[i,j]) for (i,j) in edges)

    # solves semidefinite program, optimizes linear cost function
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()

    # normalizes matrix, makes it applicable in unit sphere
    sqrtProb = scipy.linalg.sqrtm(x.value)

    #generates random hyperplane used to split set of points into two disjoint sets of nodes
    hyperplane = np.random.randn(n)

    #gives value -1 if on one side of plane and 1 if on other
    #returned as a array
    sqrtProb = np.sign( sqrtProb @ hyperplane)
    print(sqrtProb)

    colors = ["r" if sqrtProb[i] == -1 else "c" for i in range(n)]
    solution = [0 if sqrtProb[i] == -1 else 1 for i in range(n)]

    pos = nx.spring_layout(graph)
    # draw_graph(graph, colors, pos)
    obj = obj_maxcut(solution, graph)
    print("obj: ", obj, ",solution = " + str(solution))
    return solution

if __name__ == '__main__':
    # n = 5
    # graph = nx.Graph()
    # graph.add_nodes_from(np.arange(0, 4, 1))
    #
    # edges = [(1, 2), (1, 3), (2, 4), (3, 4), (3, 0), (4, 0)]
    # # edges = [(0,1),(1,2),(2,3),(3,4)]#[(1,2),(2,3),(3,4),(4,5)]
    # graph.add_edges_from(edges)

    # colors = ["g" for node in G.nodes()]
    # pos = nx.spring_layout(graph)
    # draw_graph(G, colors, pos)
    # w = np.zeros([n, n])
    # for i in range(n):
    #     for j in range(n):
    #         temp = graph.get_edge_data(i, j, default=0)
    #         if temp != 0:
    #             w[i, j] = 1

    # graph = read_nxgraph('../data/syn/syn_50_176.txt')
    graph = read_nxgraph('../data/gset/gset_14.txt')
    goemans_williamson(graph)
