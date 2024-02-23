from utils.constants import *
from sklearn.model_selection import train_test_split
from utils.general_utility_functions import *
from utils.generate_train_test import *
import networkx as nx
import numpy as np
from random import sample
import random
from sklearn.model_selection import train_test_split
import pickle
import argparse


def run_independent_set(neighbor_lambda, input_G, tau_number, tau_mut, tau_degree, seed = None, debug=False, random_start = True):
    total_deleted = 0
    
    if seed is not None:
        random.seed(seed)
    
    G = input_G.copy()
    
    node_to_score = {}
    
    #First step is to calculate node weight for every node, need to get max values
    largest_number_isolates = 0
    largest_mut_score = 0
    largest_degree = 0
 
    for node,attributes in G.nodes(data=True):
        if attributes["number_isolates"] > largest_number_isolates:
            largest_number_isolates = attributes["number_isolates"]

        if attributes["mutational_score"] > largest_mut_score:
            largest_mut_score = attributes["mutational_score"]
        
        if G.degree[node] > largest_degree:
            largest_degree = G.degree[node]
    
    #Now we calculate the scores
    for node, attributes in G.nodes(data = True):
        number_score = attributes["number_isolates"]/largest_number_isolates
        mut_score = 1 - (attributes["mutational_score"]/largest_mut_score)
        degree_score = 1 - (G.degree[node]/largest_degree)
        
        node_to_score[node] = (1/3)*(tau_number*number_score+tau_mut*mut_score+tau_degree*degree_score)
    
    #Now we calculated the score for every node, we go to run the algorithm
    independent_set = []
    
    iterations = 0
    
    while not nx.is_empty(G):
        if not random_start:
            chosen_node = max(node_to_score, key=node_to_score.get)
        else:
            if random.random() > iterations/1000:
                chosen_node = sample(list(G.nodes()), 1)[0]
            else:
                chosen_node = max(node_to_score, key=node_to_score.get)

        del node_to_score[chosen_node]
        
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
            total_isolates_deleted = 0
            
            for i in neighbors_to_delete:
                num = G.nodes[i]["number_isolates"]
                total_isolates_deleted += num
                total_deleted += num
            
            print(f"This iteration deleted {total_isolates_deleted} samples from neighbors")
            print(f"Total deleted now at {total_deleted}")
            
        for neighbor in neighbors_to_delete:
            G.remove_node(neighbor)
            del node_to_score[neighbor]
        
        G.remove_node(chosen_node)
    
        iterations += 1
    
    independent_set.append('NO_MUTATION')
    if debug:
        print(f"Total deleted {total_deleted}")
    return independent_set


def generate_mutational_split(
	number_splits:		int, 
	drug_graph:		object, 
	drug:			str,
	barcode_file:		str,
	tau_number:		float,
	tau_mut:		float,
	tau_degree:		float,
	):

        if 'covid' in barcode_file or 'gfp' in barcode_file or 'spike' in barcode_file:
                mapping = False
        else:
                mapping = True

        if drug != 'covid' and drug != 'gfp' and drug != 'spike-covid':
                all_strains_present = Generate_Train_Test(None, None, ALIGNMENT_LOCATION, False, drug).return_isolates_with_data()

        for lambda_param in np.arange(0, 1.05, 0.05):
                lambda_param = float(str(lambda_param)[:4])
                print(lambda_param)
                for i in range(number_splits):
                        print(f"Tau number {tau_number}, Tau mut {tau_mut}, Tau degree {tau_degree}")
                        independent_set = run_independent_set(lambda_param**2, drug_graph, tau_number, tau_mut, tau_degree, random_start = False)
                        print(f"We have {len(independent_set)} barcodes")
                        if 'spike' not in drug: 
                                train_barcodes, test_barcodes = split_barcodes_subset_sum(independent_set, barcode_file)
                        else:
                                train_barcodes, test_barcodes = train_test_split([i for i in independent_set if 'NO' not in i], test_size=0.2)              
                        train_strains = return_isolates_barcode(train_barcodes, barcode_file, mapping)
                        test_strains = return_isolates_barcode(test_barcodes, barcode_file, mapping)
                        print(f"In total we have {len(train_strains) + len(test_strains)} number isolates")
                        if drug == 'covid' or drug == 'gfp' or 'spike' in drug:
                                data_generation = Generate_Train_Test_GENERAL(train_strains, test_strains, drug)
                                data_generation.generate_train_test(f'{drug}_{lambda_param}_MUTATION_SPLIT_{i}')
                        else:
                                data_generation = Generate_Train_Test(train_strains, test_strains, ALIGNMENT_LOCATION, file_provided = False)
                                data_generation.generate_train_test(f'{drug}_{lambda_param}_MUTATION_SPLIT_{i}', drug, all_strains_present)

if __name__ == "__main__":
        parser = argparse.ArgumentParser("Run Independent Set Algorithm")
        parser.add_argument("drug", help="Which drug to run on", type=str)
        parser.add_argument("tau_number", help="weight of number of strain term in node weighing", type=float)
        parser.add_argument("tau_mut", help="weight of mutational frequency term in node weighing",type=float)
        parser.add_argument("tau_degree", help="weight of degree term in node weiging", type = float)
        parser.add_argument("number_splits", help="number splits per lambda", type=int)
        args = parser.parse_args()

        graph_to_input = f'{GENERAL_DATA_PATH}comutational_graph/{args.drug}_comutational.graph'

        G = pickle.load(open(graph_to_input, 'rb'))

        if args.drug == 'covid':
                barcode_file = COVID_BARCODES
        elif args.drug == 'spike-covid':
                barcode_file = COVID_SPIKE_BARCODES 
        elif args.drug == 'gfp':
                barcode_file = GFP_BARCODES
        else:		
                barcode_file = f'{GENERAL_DATA_PATH}mutational_barcodes_reg_{args.drug}'

        params_split_generation = {
        "number_splits":args.number_splits,
        "drug_graph": G,
        "drug": args.drug,
        "tau_number": args.tau_number,
        "tau_mut": args.tau_mut,
        "tau_degree": args.tau_degree,
	    "barcode_file": barcode_file
        }

        generate_mutational_split(**params_split_generation)



