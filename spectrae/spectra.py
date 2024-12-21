import networkx as nx
from .independent_set_algo import run_independent_set
from sklearn.model_selection import train_test_split
import os 
import pickle
from .utils import Spectral_Property_Graph
from .dataset import SpectraDataset
import numpy as np
from tqdm import tqdm
import pandas as pd
from abc import ABC, abstractmethod
import pickle
import torch
from typing import List, Tuple, Optional, Dict

class Spectra(ABC):

    def __init__(self, dataset: SpectraDataset, 
                 spg: Spectral_Property_Graph):
        #SPECTRA properties should be a function that given two samples in your dataset, returns whether they are similar or not
        #Cross split overlap should be a function that given two lists of samples, returns the overlap between the two lists
        self.dataset = dataset
        self.SPG = spg
        self.binary = self.SPG.binary
    
    @abstractmethod
    def spectra_properties(self, sample_one, sample_two):
        """
        Define the Spectral property between two samples
        sample_one: a single sample as defined by the __getitem__ method of the dataset
        sample_two: a single sample as defined by the __getitem__ method of the dataset
        """
        pass

    def cross_split_overlap(self, 
                            split: List[int],
                            split_two: Optional[List[int]] = None) -> Tuple[float, float, float]:
        
        def calculate_overlap(index_to_gather):
            if self.SPG.binary:
                num_similar = sum(1 for i, j in index_to_gather if self.SPG.get_weight(i, j) > 0)
                return num_similar / len(split), num_similar, len(split)
            else:
                if len(index_to_gather) > 100000000:
                    values = self.SPG.get_weights(index_to_gather)
                    return torch.mean(values).item(), torch.std(values).item(), torch.max(values).item(), torch.min(values).item()
                index_to_gather = torch.tensor(index_to_gather).cuda()
                values = self.SPG.get_weights(index_to_gather)
                return torch.mean(values).item(), torch.std(values).item(), torch.max(values).item(), torch.min(values).item()
        
        if split_two is None:
            index_to_gather = [(split[i], split[j]) for i in range(len(split)) for j in range(i + 1, len(split))]
        else:
            index_to_gather = [(split[i], split_two[j]) for i in range(len(split)) for j in range(len(split_two))]
        
        return calculate_overlap(index_to_gather)

    def return_spectra_graph_stats(self):
        num_nodes, num_edges, density = self.SPG.stats()
        print("Stats for SPECTRA property graph (SPG)")
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")
        print(f"Density of SPG: {density}")
        return num_nodes, num_edges, density
    
    def spectra_train_test_split(self, nodes, test_size, random_state):
        train = []
        test = []
        to_add = []

        for i in nodes:
            if type(i) is list:
                train.extend(i)
            else:
                to_add.append(i)
        
        tr, te = train_test_split(to_add, test_size=test_size, random_state=random_state)
        train.extend(tr)
        test.extend(te)
        return train, test

    def get_samples(self, nodes):
        return [self.dataset[i] for i in nodes]
    
    def get_sample_indices(self, samples):
        return [self.dataset.index(i) for i in samples]

    def generate_spectra_split(self, 
                               spectral_parameter: float, 
                               random_seed: int = 42, 
                               test_size: float = 0.2, 
                               degree_choosing: bool = False, 
                               minimum: int = None):
        
        print(f"Generating SPECTRA split for spectral parameter {spectral_parameter} and dataset {self.dataset.name}")
        result = run_independent_set(spectral_parameter, self.SPG,
                                seed = random_seed,
                                binary = self.binary, 
                                minimum = minimum,
                                degree_choosing = degree_choosing)

        if len(result) <= 5:
            return None, None, None
        print(f"Number of samples in independent set: {len(result)}")
        train, test = self.spectra_train_test_split(result, test_size=test_size, random_state=random_seed)
        stats = self.get_stats(train, test, spectral_parameter)
        return train, test, stats
    
    def get_stats(self, train, test, spectral_parameter):
        train_size = len(train)
        test_size = len(test)
        if not self.binary:
            cross_split_overlap, std_css, max_css, min_css = self.cross_split_overlap(self.get_sample_indices(train), self.get_sample_indices(test))
            stats = {'SPECTRA_parameter': spectral_parameter, 
                    'train_size': train_size, 
                    'test_size': test_size, 
                    'cross_split_overlap': cross_split_overlap,
                    'std_css': std_css,
                    'max_css': max_css,
                    'min_css': min_css}
        else:
            cross_split_overlap, num_similar, num_total = self.cross_split_overlap(self.get_sample_indices(train))
            stats = {'SPECTRA_parameter': spectral_parameter, 
                    'train_size': train_size, 
                    'test_size': test_size, 
                    'cross_split_overlap': cross_split_overlap,
                    'num_similar': num_similar,
                    'num_total': num_total}
        return stats
    
    def generate_spectra_splits(self, 
                                spectral_parameters: List[float], 
                                number_repeats: int, 
                                random_seed: List[float], 
                                test_size: float = 0.2,
                                degree_choosing: bool = False,
                                minimum: int = None,
                                force_reconstruct: bool = False,
                                path_to_save: str = None):
        
        #Random seed is a list of random seeds for each number
        name = self.dataset.name
        if self.binary:
            if self.SPG.get_density() >= 0.4:
                raise Exception("Density of SPG is greater than 0.4, SPECTRA will not work as your dataset is too similar to itself. Please check your dataset and SPECTRA properties.")

        if path_to_save is None:
            path_to_save = f"{name}_SPECTRA_splits"
        
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        splits = []
        for spectral_parameter in spectral_parameters:
            for i in range(number_repeats):
                if os.path.exists(f"{path_to_save}/SP_{spectral_parameter}_{i}") and not force_reconstruct:
                    print(f"Folder SP_{spectral_parameter}_{i} already exists. Skipping")
                elif force_reconstruct or not os.path.exists(f"{path_to_save}/SP_{spectral_parameter}_{i}"):
                    train, test, stats = self.generate_spectra_split(float(spectral_parameter), random_seed[i], test_size, degree_choosing, minimum)
                    if train is not None:
                        if not os.path.exists(f"{path_to_save}/SP_{spectral_parameter}_{i}"):
                            os.makedirs(f"{path_to_save}_SPECTRA_splits/SP_{spectral_parameter}_{i}")
                
                        pickle.dump(train, open(f"{path_to_save}_SPECTRA_splits/SP_{spectral_parameter}_{i}/train.pkl", "wb"))
                        pickle.dump(test, open(f"{path_to_save}_SPECTRA_splits/SP_{spectral_parameter}_{i}/test.pkl", "wb"))
                        pickle.dump(stats, open(f"{path_to_save}_SPECTRA_splits/SP_{spectral_parameter}_{i}/stats.pkl", "wb"))
                    else:
                        print(f"Split for SP_{spectral_parameter}_{i} could not be generated since independent set only has one sample")
                
        return splits
    
    def return_split_stats(self, spectral_parameter: float, 
                           number: int, 
                           path_to_save: str = None):
        
        if path_to_save is None:
            path_to_save = f"{self.dataset.name}_SPECTRA_splits"

        split_folder = f"./{path_to_save}/SP_{spectral_parameter}_{number}"
        if not os.path.exists(split_folder):
            raise Exception(f"Split folder {split_folder} does not exist")
        else:
            if not os.path.exists(f"{split_folder}/stats.pkl"):
                train = pickle.load(open(f"{split_folder}/train.pkl", "rb"))
                test = pickle.load(open(f"{split_folder}/test.pkl", "rb"))
                stats = self.get_stats(train, test, spectral_parameter)
                pickle.dump(stats, open(f"{split_folder}/stats.pkl", "wb"))
                return stats
            
            return pickle.load(open(f"{split_folder}/stats.pkl", "rb"))
    
    def return_split_samples(self, spectral_parameter: float, 
                             number: int,
                             path_to_save: str = None):
        
        if path_to_save is None:
            path_to_save = f"{self.dataset.name}_SPECTRA_splits"

        split_folder = f"./{path_to_save}/SP_{spectral_parameter}_{number}"
        if not os.path.exists(split_folder):
            raise Exception(f"Split folder {split_folder} does not exist")
        else:
            train = pickle.load(open(f"{split_folder}/train.pkl", "rb"))
            test = pickle.load(open(f"{split_folder}/test.pkl", "rb"))
            return [self.dataset[int(i)] for i in train], [self.dataset[int(i)] for i in test]
    
    def return_all_split_stats(self,
                               path_to_save: str = None,
                               show_progress: bool = False) -> Dict:
        
        if path_to_save is None:
            path_to_save = f"{self.dataset.name}_SPECTRA_splits"

        SP = []
        numbers = []
        train_size = []
        test_size = []
        cross_split_overlap = []

        if not show_progress:
            to_iterate = os.listdir(path_to_save)
        else:
            to_iterate = tqdm(os.listdir(path_to_save))

        for folder in to_iterate:
            spectral_parameter = folder.split('_')[1]
            number = folder.split('_')[2]
            res = self.return_split_stats(spectral_parameter, number)
            SP.append(float(spectral_parameter))
            numbers.append(int(number))
            train_size.append(int(res['train_size']))
            test_size.append(int(res['test_size']))
            cross_split_overlap.append(float(res['cross_split_overlap']))
        
        stats = {'SPECTRA_parameter': SP, 'number': number, 'train_size': train_size, 'test_size': test_size, 'cross_split_overlap': cross_split_overlap}
        return stats

class Spectra_Property_Graph_Constructor():
    def __init__(self, spectra: Spectra, 
                 dataset: SpectraDataset,
                 num_chunks: int = 0):
        self.spectra = spectra
        self.dataset = dataset
        if num_chunks != 0:
            self.data_chunk = np.array_split(list(range(len(self.dataset))), num_chunks)
        else:
            self.data_chunk = [list(range(len(self.dataset)))]
    
    def create_adjacency_matrix(self, chunk_num: int):
        to_store = []

        for i in tqdm(self.data_chunk[chunk_num]):
            for j in range(i, len(self.dataset)):
                if i != j:
                    if self.spectra.spectra_properties(self.dataset[i], self.dataset[j]):
                        to_store.append(1)
                    else:
                        to_store.append(0)
        
        with open(f'adjacency_matrices/aj_{chunk_num}.npy', 'wb') as f:
            pickle.dump(to_store, f)
        
    def combine_adjacency_matrices(self):
        num_adjacency = len(os.listdir('adjacency_matrices'))
        if self.num_chunks == 0:
            if num_adjacency != 1:
                raise Exception("Need to generate adjacency matrices first! See documentation")
        else:
            if num_adjacency != self.num_chunks:
                raise Exception("Need to generate adjacency matrices first! See documentation")
        
        n = len(self.dataset)
        new = np.zeros((n*(n-1))/2)
        previous_start = 0

        for i in tqdm(range(self.num_chunks)):
            to_assign = np.load(f'aj_{i}.npy', allow_pickle=True)
            new[previous_start:previous_start+len(to_assign)] = to_assign
            previous_start += len(to_assign)
            new = new.astype(np.int8)

        if self.spectra.binary:
            torch.save(torch.tensor(new).to(torch.int8), 'flattened_adjacency_matrix.pt')
        else:
            torch.save(torch.tensor(new).half(), 'flattened_adjacency_matrix.pt')
            

    

    


    

