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
                        num_splits: int = None,
                        debug_mode: bool = False):
    


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
    if debug_mode:
        print(f"Threshold is {threshold}")
    indices_deleted = []
    full_indices_deleted = []

    expected_number_delete = int(n * spectral_parameter)
    if debug_mode:
        print(expected_number_delete)
    min_degree_node = input_G.get_minimum_degree_node()
    num_deleted_in_iteration = 0 
    
    while len(indices_to_scan) > 0:
        if debug_mode:
            print(f'Num possibly deleted {len(indices_deleted)}, num actually deleted {num_deleted_in_iteration}, number of nodes left to consider {len(indices_to_scan)}')
        num_deleted_in_iteration = 0 
        indices_deleted = []
        if degree_choosing:
            if len(full_indices_deleted) > 0:
                chosen_node, _ = min_degree_node.send(full_indices_deleted)
            else:
                chosen_node, _ = next(min_degree_node)
        else:
            chosen_node = random.sample(indices_to_scan, 1)[0]

        indices_to_scan.remove(chosen_node)
        full_indices_deleted.append(chosen_node)
        total_num_deleted += 1
        num_deleted_in_iteration += 1

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
                        num_deleted_in_iteration += 1
                        full_indices_deleted.append(i)
                else:
                    indices_to_scan.remove(i)
                    total_num_deleted += 1
                    num_deleted_in_iteration += 1
                    full_indices_deleted.append(i)
                
                if minimum is not None:
                    if n - total_num_deleted <= minimum - len(independent_set):
                        independent_set.extend(indices_to_scan)
                        return independent_set

            indices_deleted.append(chosen_node)

        indices_to_scan = list(indices_to_scan)
        pbar.update(num_deleted_in_iteration)

        if len(indices_to_scan) != 136428 - len(full_indices_deleted):
            raise Exception("Length of indices to scan is not equal to 136428 - len(full_indices_deleted), logic is not met")
    
    pbar.close()

    return independent_set
