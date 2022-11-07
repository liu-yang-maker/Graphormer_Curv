
from itertools import chain
import itertools
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from RicciFlow1 import aStarNormalize as Ricci
import time
#from utils import load_data
#from utils import nxgraph
import os
from multiprocessing import Pool, cpu_count
from collections import ChainMap

def k_hop_neighbors(G, k, queries, nodes=set()):
    if k == 0:
        return nodes
    if isinstance(queries, (int, np.int64)):
        queries = [queries]
    neighbors = [[n for n in G[a]] for a in queries]
    neighbors = chain.from_iterable(neighbors)
    neighbors = set(neighbors) - set(nodes)
    neighbors = k_hop_neighbors(G, k-1, list(neighbors), nodes|neighbors)
    return set(neighbors)

def to_edge_graph(G):
    edge_list = nx.edges(G)
    eg_nodes = list(edge_list)
    eg_edges = set()
    for n in nx.nodes(G):
        neighbors = G[n]
        clique_nodes = [tuple(sorted((n, m))) for m in neighbors]
        clique_edges = tuple(itertools.combinations(clique_nodes, 2))
        eg_edges = eg_edges | set(clique_edges)
    edge_graph_a = nx.Graph()
    edge_graph_a.add_edges_from(eg_edges)    
    # pos = nx.spring_layout(a)
    # plt.figure()
    # nx.draw(edge_graph_a, with_labels=True)
    # plt.axis('off')
    # plt.show()
    return edge_graph_a
    
    
        


class compute_ricci_locally:

    def __init__(self, G, proc=cpu_count()):
        self.G = G
        self.proc = 2

    def _compute_ricci_curvature_single_edge(self, e1, e2):
        sub_g = nx.Graph(nx.subgraph(self.G, k_hop_neighbors(self.G, 2, [e1,e2])))
        rc = Ricci(sub_g)
        edge_ricci = rc.compute_ricci_curvature_edges(edge_list=[(e1,e2)])
        return edge_ricci

    def _wrap_compute_single_edge(self, stuff):
        return self._compute_ricci_curvature_single_edge(*stuff)

    def compute_ricci_curvature_edges(self, edge_list=None, return_edge_attr=False): ######
        
        # Start compute edge Ricci curvature
        p = Pool(processes=self.proc)

        # Compute Ricci curvature for edges
        args = [(source, target) for source, target in edge_list]

        #result = list(map(self._wrap_compute_single_edge, args))
        result = p.map_async(self._wrap_compute_single_edge, args).get()
        p.close()
        p.join()
        res = ChainMap(*result)

        nx.set_edge_attributes(self.G, dict(res), 'ricciCurvature')

        if return_edge_attr:
            return dict(res)

# a = nx.Graph()
# edge_list = [(1,2), (1,3), (2,3), 
#              (2,4), 
#              (4,5), (5,6), (6,7), (4,7)]
# a.add_edges_from(edge_list)
# # pos = nx.spring_layout(a)

# time1_start = time.time()
# rc_a = Ricci(a)
# rc_a.compute_ricci_curvature()
# print('time_1:', time.time()-time1_start)
# print({(x,y):'%.4f'%z for (x,y),z in nx.get_edge_attributes(rc_a.G, 'ricciCurvature').items()})

# edge_graph_a = to_edge_graph(a)
# time2_start = time.time()
# ricci_G = compute_ricci_locally(a)
# ricci_G.compute_ricci_curvature_edges(a.edges())
# print('time_2:', time.time()-time2_start)
# print({(x,y):'%.4f'%z for (x,y),z in nx.get_edge_attributes(ricci_G.G, 'ricciCurvature').items()})


# print('Cora!')
# path = os.path.join(os.getcwd(), 'data')
# data = load_data(path, 'Cora')
# a = nxgraph(data.edge_index)

# time1_start = time.time()
# rc_a = Ricci(a)
# rc_a.compute_ricci_curvature()
# print('time_1:', time.time()-time1_start)
# print({(x,y):'%.4f'%z for (x,y),z in nx.get_edge_attributes(rc_a.G, 'ricciCurvature').items()})

# time2_start = time.time()

# ricci_G = compute_ricci_locally(a)
# ricci_G.compute_ricci_curvature_edges(a.edges())
# print('time_2:', time.time()-time2_start)
# print({(x,y):'%.4f'%z for (x,y),z in nx.get_edge_attributes(ricci_G.G, 'ricciCurvature').items()})