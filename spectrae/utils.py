import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Optional
import os
import matplotlib.pyplot as plt

class FlattenedAdjacency:
    def __init__(self, 
                 flattened_adjacency_path: str):
        
        self.flattened_adjacency = torch.load(flattened_adjacency_path)[:24929323695]
        if torch.cuda.is_available():
            self.flattened_adjacency = self.flattened_adjacency.cuda()
        self.n = self.get_number_len(len(self.flattened_adjacency))
        if self.flattened_adjacency.dtype is torch.int8:
            self.binary = True
        elif self.flattened_adjacency.dtype is torch.bfloat8:
            self.binary = False
        else:
            raise ValueError("Invalid datatype. Use torch.int8 or torch.bfloat16.")
    
    def get_number_len(self, number_items):
        return int(abs(-1-np.sqrt(1+8*number_items))/2)
    
    def return_index_flat(self, i, j):
        if i == j:
            return 1

        if i > j:
            i, j = j, i
            
        sum  = i * self.n - (i * (i + 1)) // 2
        return sum + j - i - 1

    def __len__(self):
        return self.n
    
    def __getitem__(self, indices):
        if isinstance(indices, tuple) and len(indices) == 2:
            i, j = indices
            return self.flattened_adjacency[self.return_index_flat(i, j)].to(torch.int64)
        elif isinstance(indices, list):
            to_index = []
            for i, j in indices:
                to_index.append(self.return_index_flat(i, j))
            return self.flattened_adjacency[to_index].to(torch.int64)
        else:
            raise IndexError("Invalid index. Use FlattenedAdjacency[i, j] or FlattenedAdjacency[[i,j],[k,l]] for indexing.")

class Spectral_Property_Graph:
    def __init__(self, 
                 flattened_adjacency: FlattenedAdjacency):
        
        self.flattened_adjacency = flattened_adjacency
        self.degree_distribution = None
        self.binary = self.flattened_adjacency.binary
    
    def num_nodes(self):
        return len(self.flattened_adjacency)
    
    def num_edges(self):
        num_nodes = self.num_nodes()
        return int(num_nodes*(num_nodes-1)/2)

    def chunked_sum(self, chunk_size: int):
        """
        Sums a tensor in chunks to avoid OOM errors.
        
        Args:
            tensor (torch.Tensor): The input tensor to be summed.
            chunk_size (int): The size of each chunk.
        
        Returns:
            torch.Tensor: The sum of the tensor.
        """
        tensor = self.flattened_adjacency.flattened_adjacency
        total_sum = torch.zeros_like(tensor[0], dtype=torch.int64)
        num_chunks = (tensor.size(0) + chunk_size - 1) // chunk_size  # Calculate the number of chunks

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, tensor.size(0))
            chunk = tensor[start_idx:end_idx]
            total_sum += chunk.sum(dim=0)

        return total_sum

    def get_density(self, chunk_size = 1000000000, return_sum = False):
        result = self.chunked_sum(chunk_size)
        if return_sum:
            return result, result/self.num_edges()
        return result/self.num_edges()

    def get_degree(self, node):
        n = self.num_nodes()

        indexes = []
        for i in range(n):
            indexes.append(self.flattened_adjacency.return_index_flat(i, node))
            #values.append(self.flattened_adjacency[i, node])
        
        values = self.flattened_adjacency.flattened_adjacency[indexes].type(torch.int64)
        return torch.sum(values), torch.sum(values)/n
        #return sum(values), sum(values)/n

    def get_degree_distribution(self, 
                                track: bool = False, 
                                save: bool = False, 
                                name: str = "degree_distribution",
                                ) -> Dict[int, int]:
        
        if f"{name}.pt" in os.listdir():
            if not torch.cuda.is_available():
                self.degree_distribution = torch.load(f"{name}.pt", map_location = "cpu")
            self.degree_distribution = torch.load(f"{name}.pt", map_location = "cuda")
            return self.degree_distribution
        
        n = self.num_nodes()

        degrees = {}
        if track:
            to_iterate = tqdm(range(n))
        else:
            to_iterate = range(n)

        for i in to_iterate:
            degrees[i] = self.get_degree(i)[0]

        if save:
            torch.save(degrees, f"{name}.pt")

        self.degree_distribution = degrees
        return self.degree_distribution
    
    def get_minimum_degree_node(self, to_include: List[int] = []) -> Tuple[int, int]:
        if self.degree_distribution is None:
            self.degree_distribution = self.get_degree_distribution(track = True, save = True)
        
        minimum_value = None 
        minimum_key = None

        for i in to_include:
            if minimum_value is None or self.degree_distribution[i] < minimum_value:
                minimum_value = self.degree_distribution[i]
                minimum_key = i
        return minimum_key, minimum_value
        # filtered_degrees = {k: v for k, v in self.degree_distribution.items() if k not in exclude}
        # min_key = min(filtered_degrees, key=filtered_degrees.get)
        #return min_key, filtered_degrees[min_key]
    
    def get_weight(self, i, j):
        return self.flattened_adjacency[i, j]

    def get_weights(self, indices):
        return self.flattened_adjacency[indices]
    
    def get_stats(self):
        return self.num_nodes, self.num_edges, self.get_density(return_sum = False)


        
def cross_split_overlap(split, g):
    binary = g.binary
    if binary:
        num_similar = 0
        for i in range(len(split)):
            for j in range(i+1, len(split)):
                if g.get_weight(split[i], split[j]) > 0:
                    num_similar += 1
                    break
        return num_similar/len(split), num_similar, len(split)
    else:
        index_to_gather = []
        for i in range(len(split)):
            for j in range(i+1, len(split)):
                index_to_gather.append((split[i], split[j]))
            if len(index_to_gather) > 100000000:
                values = g.get_weights(index_to_gather)
                return torch.mean(values).item(), torch.std(values).item(), torch.max(values).item(), torch.min(values).item()

        index_to_gather = torch.tensor(index_to_gather).cuda()
        values = g.get_weights(index_to_gather)
        return torch.mean(values).item(), torch.std(values).item(), torch.max(values).item(), torch.min(values).item()

def output_split_stats(split_directory, g):
    spectral_parameter = []
    length = []
    css = []

    for split_file in tqdm(os.listdir(split_directory)):
        x = np.load(f'{split_directory}/{split_file}')
        sp = split_file.split('_')[0]
        spectral_parameter.append(sp)
        length.append(len(x))
        css.append(cross_split_overlap(x, g)[0])

    # Convert spectral_parameter to a numeric type if necessary
    spectral_parameter = list(map(float, spectral_parameter))

    # Create the scatter plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Dataset size vs Spectral parameter
    ax1.scatter(spectral_parameter, length, color='blue')
    ax1.set_title('Dataset Size vs Spectral Parameter')
    ax1.set_xlabel('Spectral Parameter')
    ax1.set_ylabel('Dataset Size')

    # Cross split overlap vs Spectral parameter
    ax2.scatter(spectral_parameter, css, color='red')
    ax2.set_title('Cross Split Overlap vs Spectral Parameter')
    ax2.set_xlabel('Spectral Parameter')
    ax2.set_ylabel('Cross Split Overlap')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('split_stats.png')
    plt.show()
        

    return spectral_parameter, length, css
