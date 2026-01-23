import json

sol = []
cut = 0

with open("G70.txt",'r') as f:
    fline = f.readline()
    fline = fline.split()
    num_nodes, num_edges = int(fline[0]), int(fline[1])
    with open("G70_9595.txt",'r') as f2:
        for i in range(num_nodes):
            fline = f2.readline()
            fline = fline.split()
            sol.append(int(fline[0]))
    
    for i in range(num_edges):
        fline = f.readline()
        fline = fline.split()
        a, b, value = int(fline[0])-1, int(fline[1])-1, int(fline[2])
        if (sol[a]==1 and sol[b]==0) or (sol[a]==0 and sol[b]==1):
            cut += value

print(cut)

