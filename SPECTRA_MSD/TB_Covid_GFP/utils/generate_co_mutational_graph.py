from utils.general_utility_functions import *
from utils.constants import *
import networkx as nx
import numpy as np
import pandas as pd
from itertools import chain
from tqdm import tqdm
import pickle

def create_graph(barcode_file, filtered_mutations = [], mapping = True):
    barcodes = set()

    #Initialize graph and gather relevant sample barcodes
    G = nx.Graph()
    if 'covid' in barcode_file or 'gfp' in barcode_file or 'spike' in barcode_file:
        remap_barcode = lambda x: x
        mapping = False    

    barcode_to_mut_score = return_mutational_scores(barcode_file, mapping)
    
    for line in open(barcode_file, 'r').readlines():
        data = line.split('\t')
        barcodes.add(data[1].rstrip())
    
    #Initialize graph by creating a node for every barcode
    print("GOING TO INITIALIZE GRAPH")
    for barcode in barcodes:
        if barcode:
            barcode_to_interrogate = remap_barcode(barcode)           

            number_of_isolates = return_number_isolates([barcode_to_interrogate], barcode_file, mapping)
            mut_score = barcode_to_mut_score[barcode_to_interrogate]
            G.add_node(barcode_to_interrogate)

    #Add node attributes to graph
    print("GRAPH INITIALIZED")
    node_attributes = {}
    
    for node in G.nodes:
        
        number_of_isolates = return_number_isolates([node], barcode_file, mapping)
        mut_score = barcode_to_mut_score[node]
        
        node_attributes[node] = {"number_isolates": number_of_isolates,
                                "mutational_score": mut_score}
        
    nx.set_node_attributes(G, node_attributes)
    
    filtered_mutations = [remap_barcode(i) for i in filtered_mutations] 
    print(filtered_mutations)

    #Add edges to graph
    for bar_1 in tqdm(barcodes, total=len(barcodes)):
        for bar_2 in barcodes:
            if bar_1 != bar_2 and bar_1 and bar_2:
                initial_mutational_overlap_score = sum([mut in remap_barcode(bar_2) for mut in remap_barcode(bar_1).split('-') if mut not in filtered_mutations])
                if initial_mutational_overlap_score:
                    G.add_edge(remap_barcode(bar_1), remap_barcode(bar_2))                  

    return G

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Create Co Mutational Graph")
    parser.add_argument("drug", help="Drug To Create Mutational Graph For", type = str)
    args = parser.parse_args()

    mapping = True

    if args.drug == 'covid':
        barcode_file = COVID_BARCODES
        mapping = False
    elif args.drug == 'spike-covid':
        barcode_file = COVID_SPIKE_BARCODES
        mapping = False
    elif args.drug == 'gfp':
        barcode_file = GFP_BARCODES
        mapping = False
    else: 
        barcode_file = f'{GENERAL_DATA_PATH}mutational_barcodes_reg_{args.drug}'

    G = create_graph(barcode_file, [], mapping)
    graph_output = f'{GENERAL_DATA_PATH}comutational_graph/{args.drug}_fil_comutational.graph'
    pickle.dump(G, open(graph_output, 'wb'))





