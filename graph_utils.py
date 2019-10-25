""" This module implements graph operations that are not 
    implemented by networkx"""


import networkx as nx
import numpy as np

def threshold_graph(G, keep_weights=True):

    # Computing threshold value 
    edge_data = G.edges.data()
    original_weights = [d[-1]['weight'] for d in edge_data]

    # one std above median
    min_weight = np.median(original_weights) + 3*np.std(original_weights)

    Gnew = nx.Graph()
    for (u,v,w) in G.edges(data=True):
        if w['weight'] > min_weight :
            if keep_weights:
                new_w = w['weight']
            else:
                new_w = 1
            Gnew.add_edge(u, v, weight=new_w)
    return Gnew

def degree_dist_entropy(G):
    degree_sequence = [d for n, d in G.degree()]
    hist = np.histogram(degree_sequence, bins=len(G.nodes))
    hist_values = hist[0]
    p_k = hist_values[hist_values != 0] / len(G.nodes)
    return -sum(p_k * np.log10(p_k)) 
        