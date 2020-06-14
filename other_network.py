import numpy as np 
import networkx as nx 
import igraph
filename = '../celegansneural.gml'
a = igraph.Graph.Read_GML(filename)
data = nx.read_gml(filename)
A = nx.adjacency_matrix(data).todense()
