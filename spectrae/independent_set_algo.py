import random
import numpy as np
from tqdm import tqdm
import torch
from .utils import FlattenedAdjacency, Spectral_Property_Graph, cross_split_overlap

def run_independent_set(spectral_parameter: int, 
                        input_G: Spectral_Property_Graph,
                        seed: int = 42,
                        binary: bool = True,
                        minimum: int = None,
                        degree_choosing: bool = False,
                        num_splits: int = None):
    


    total_num_deleted = 0
    independent_set = []
    random.seed(seed)

    n = input_G.num_nodes()
    indices_to_scan = list(range(n))
    if spectral_parameter == 0:
        return indices_to_scan
    pbar = tqdm(total = len(indices_to_scan))

    #Trying  a non-percentile approach
    #Note this assumes there are 20 
    if not binary:
        if num_splits is None:
            raise Exception("Num splits must be specified for non-binary graphs, see documentation for more information")
        threshold = spectral_parameter*(torch.max(input_G) - torch.min(input_G))/num_splits 
    else:
        threshold = 0 
    print(f"Threshold is {threshold}")
    indices_deleted = []

    expected_number_delete = int(n * spectral_parameter)
    print(expected_number_delete)
    
    while len(indices_to_scan) > 0:
        print(len(indices_deleted))
        indices_deleted = []
        if degree_choosing:
            chosen_node, _ = input_G.get_minimum_degree_node(indices_to_scan)
        else:
            chosen_node = random.sample(indices_to_scan, 1)[0]

        indices_to_scan.remove(chosen_node)

        to_iterate = indices_to_scan[:]
        
        indices_to_gather = []

        for index in to_iterate:
            indices_to_gather.append((chosen_node, index))

        values = input_G.get_weights(indices_to_gather)

        indices_deleted.extend(list(torch.tensor(to_iterate).cuda()[values > threshold].cpu().numpy()))

        indices_deleted = list(set(indices_deleted))
        indices_to_scan = set(indices_to_scan)
        
        if len(indices_deleted) > expected_number_delete:
            indices_deleted = [chosen_node]
            total_num_deleted += 1
        else:
            independent_set.append(chosen_node)
            for i in indices_deleted:
                if binary:
                    if random.random() < spectral_parameter:
                        indices_to_scan.remove(i)
                        total_num_deleted += 1
                else:
                    indices_to_scan.remove(i)
                    total_num_deleted += 1
                
                if minimum is not None:
                    if n - total_num_deleted <= minimum - len(independent_set):
                        independent_set.extend(indices_to_scan)
                        return independent_set

            indices_deleted.append(chosen_node)
        indices_to_scan = list(indices_to_scan)
        pbar.update(len(indices_deleted))

    
    pbar.close()

    return independent_set
