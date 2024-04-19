import pandas as pd
import random
import pickle
similarity = pickle.load(open('protein_similarity', 'rb'))
import networkx as nx
G = nx.read_gexf("homology_graph.gexf")
print("Nice! Loaded everything")
def run_independent_set(neighbor_lambda, input_G, tau_number, tau_mut, tau_degree, seed = None, debug=False, random_start = True):
    total_deleted = 0
     
    if seed is not None:
        random.seed(seed)
    
    G = input_G.copy()
    
    #Now we calculated the score for every node, we go to run the algorithm
    independent_set = []
    
    iterations = 0
    
    while not nx.is_empty(G):
        if not random_start:
            chosen_node = max(node_to_score, key=node_to_score.get)
        else:
            chosen_node = sample(list(G.nodes()), 1)[0]
        
        independent_set.append(chosen_node)
        neighbors = G.neighbors(chosen_node)
        neighbors_to_delete = []
        
        for neighbor in neighbors:
            if neighbor_lambda == 1.0:
                neighbors_to_delete.append(neighbor)
            elif neighbor_lambda != 0.0:
                if random.random() < neighbor_lambda:
                    neighbors_to_delete.append(neighbor)

        if debug:
            print(f"Iteration {iterations} Stats")
            print(f"Deleted {len(neighbors_to_delete)} nodes")

        for neighbor in neighbors_to_delete:
            G.remove_node(neighbor)
        
        if chosen_node not in neighbors_to_delete:
            G.remove_node(chosen_node)
    
        iterations += 1

    if debug:
        print(f"Total deleted {total_deleted}")
    return independent_set

def calculate_proportions(x_train, x_test):
    num_similar = 0
    
    for i in x_test:
        for j in x_train:
            try:
                if similarity[(i, j)] > 0.3:
                   num_similar += 1
                   break
            except Exception:
                if similarity[(j, i)] > 0.3:
                   num_similar += 1
                   break
    
    return num_similar/len(x_test)

import numpy as np
from random import sample
from sklearn.model_selection import train_test_split

calculated_proportions = []
lambda_params = []
number_samples = []
family_split = []
lambda_to_splits = {}
    
print("Beginning Scan")
for lambda_param in np.arange(0, 1.025, 0.025):
    lambda_params.append(lambda_param)
    result = run_independent_set(lambda_param, G, 0, 0, 0)
    x_train ,x_test = train_test_split(result,test_size=0.2)
    proportion = calculate_proportions([int(i) for i in x_train], [int(i) for i in x_test])
    calculated_proportions.append(proportion)
    lambda_to_splits[lambda_param] = [x_train, x_test]
    print(f"{lambda_param}\t{proportion}\t{len(result)}")
    number_samples.append(len(result))

    print(f"{lambda_param}\t{proportion}\t{len(result)}")
    result_file = open(f"homology_splits/lambda_{lambda_param}.txt", "w")
    result_file.write(f"{lambda_param}\t{proportion}\t{len(result)}\n")
    result_file.write(f"Train: {x_train}\n")
    result_file.write(f"Test: {x_test}\n")
    result_file.close()

pickle.dump(lambda_param, open('lambda_param', 'wb'))
pickle.dump(calculated_proportions, open('calculated_proportions', 'wb'))
pickle.dump(number_samples, open('number_samples', 'wb'))


