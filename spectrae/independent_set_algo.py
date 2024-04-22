import random 
import networkx as nx
from .utils import is_clique, connected_components, is_integer
from scipy import stats
import numpy as np

def run_independent_set(spectral_parameter, input_G, seed = None, 
                        debug=False, distribution  = None, binary = True):
    total_deleted = 0
    independent_set = []
    
    if seed is not None:
        random.seed(seed)
    
    G = input_G.copy()

    if binary:
        #First check if any connected component of the graph is a clique, if so, add it as one unit to the independent set
        components = list(connected_components(G))
        deleted = 0
        for i, component in enumerate(components):
            subgraph = G.subgraph(component)
            if is_clique(subgraph):
                print(f"Component {i} is too densly connected, adding samples as a single unit to independent set and deleting them from the graph")
                independent_set.append(list(subgraph.nodes()))
                G.remove_nodes_from(subgraph.nodes())
            else:
                for node in list(subgraph.nodes()):
                    if subgraph.degree(node) == len(subgraph.nodes()) - 1:
                        deleted += 1
                        G.remove_node(node)
        
        print(f"Deleted {deleted} nodes from the graph since they were connected to all other nodes")

    iterations = 0
    
    while not nx.is_empty(G):
        chosen_node = random.sample(list(G.nodes()), 1)[0]
        
        independent_set.append(chosen_node)
        neighbors = G.neighbors(chosen_node)
        neighbors_to_delete = []
        
        for neighbor in neighbors:
            if not binary:
                if spectral_parameter == 1.0:
                    neighbors_to_delete.append(neighbor)
                else:
                    edge_weight = G[chosen_node][neighbor]['weight']
                    if distribution is None:
                        raise Exception("Distribution must be provided if binary is set to False, must precompute similarities")
                    if random.random() < spectral_parameter and (1-spectral_parameter)*100 < stats.percentileofscore(distribution, edge_weight):
                        neighbors_to_delete.append(neighbor)
            else:
                if spectral_parameter == 1.0:
                    neighbors_to_delete.append(neighbor)
                elif spectral_parameter != 0.0:
                    if random.random() < spectral_parameter:
                        neighbors_to_delete.append(neighbor)

        if debug:
            print(f"Iteration {iterations} Stats")
            print(f"Deleted {len(neighbors_to_delete)} nodes from {G.degree(chosen_node)} neighbors of node {chosen_node}")
            total_deleted += len(neighbors_to_delete)
            
        for neighbor in neighbors_to_delete:
            G.remove_node(neighbor)
        
        if chosen_node not in neighbors_to_delete:
            G.remove_node(chosen_node)
    
        iterations += 1
    
    for node in list(G.nodes()):
        #Append the nodes left to G
        independent_set.append(node)

    if debug:
        print(f"{len(input_G.nodes())} nodes in the original graph")
        print(f"Total deleted {total_deleted}")
        print(f"{len(independent_set)} nodes in the independent set")

    return independent_set