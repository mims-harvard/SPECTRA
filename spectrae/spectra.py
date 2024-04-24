import networkx as nx
from .independent_set_algo import run_independent_set
from sklearn.model_selection import train_test_split
import os 
import pickle
from .utils import is_clique, connected_components, is_integer
import numpy as np
from tqdm import tqdm
import pandas as pd
from abc import ABC, abstractmethod

class Spectra(ABC):

    def __init__(self, dataset, 
                 binary = True):
        #SPECTRA properties should be a function that given two samples in your dataset, returns whether they are similar or not
        #Cross split overlap should be a function that given two lists of samples, returns the overlap between the two lists
        self.dataset = dataset
        self.SPG = None
        self.spectra_properties_loaded = None
        self.binary = binary
    
    @abstractmethod
    def spectra_properties(self, sample_one, sample_two):
        """
        Define the Spectral property between two samples
        sample_one: a single sample as defined by the __getitem__ method of the dataset
        sample_two: a single sample as defined by the __getitem__ method of the dataset
        """
        pass

    @abstractmethod
    def cross_split_overlap(self, train, test):
        """
        Define the cross split overlap between two lists of samples.
        Ideally should be a number between 0 and 1 that defines the overlap between the two lists of samples 

        """
        pass        
    
    def construct_spectra_graph(self, force_reconstruct = False):
        if self.SPG is not None:
            return self.SPG
        elif os.path.exists(f"{self.dataset.name}_spectral_property_graphs/{self.dataset.name}_SPECTRA_property_graph.gexf") and not force_reconstruct:
            print("Loading spectral property graph")
            self.SPG = nx.read_gexf(f"{self.dataset.name}_spectral_property_graphs/{self.dataset.name}_SPECTRA_property_graph.gexf")
            self.return_spectra_graph_stats()
            return self.SPG
        else:
            self.SPG = nx.Graph()
            if self.spectra_properties_loaded is not None:
                for row in tqdm(self.spectra_properties_loaded.itertuples(), 
                                total = len(self.spectra_properties_loaded)):
                    if row[3]:
                        self.SPG.add_edge(row[1], row[2], weight = row[3])
            else:
                for i in tqdm(range(len(self.dataset))):
                    for j in range(i+1, len(self.dataset)):
                        if self.spectra_properties_loaded is not None:
                            weight = self.spectra_properties_loaded[(self.spectra_properties_loaded[0] == i) & (self.spectra_properties_loaded[1] == j)][2].values[0]
                        else:    
                            weight = self.spectra_properties(self.dataset[i], self.dataset[j])
                        if weight:
                            self.SPG.add_edge(i, j, weight = weight)
            
            self.return_spectra_graph_stats()

            if self.binary:
                #Check to make sure SPG is not fully connected
                if is_clique(self.SPG):
                    print("SPG is fully connected")
                    raise Exception("The SPG is fully connected, cannot run SPECTRA, all samples are similar to each other")
                else:
                    print("SPG is not fully connected")
                    components = list(connected_components(self.SPG))
                    all_fully_connected = True
                    for i, component in enumerate(components):
                        subgraph = self.SPG.subgraph(component)
                        if is_clique(subgraph):
                            print(f"Component {i} is fully connected, all samples are similar to each other")
                            #raise Exception("The SPG is fully connected, cannot run SPECTRA")
                        else:
                            all_fully_connected = False
                            print(f"Component {i} is not fully connected")
                    if all_fully_connected:
                        raise Exception("All SPG sub components are fully connected, cannot run SPECTRA, all samples are similar to each other")
                
            if not os.path.exists(f"{self.dataset.name}_spectral_property_graphs"):
                os.makedirs(f"{self.dataset.name}_spectral_property_graphs")
            
            nx.write_gexf( self.SPG, f"{self.dataset.name}_spectral_property_graphs/{self.dataset.name}_SPECTRA_property_graph.gexf")

            return self.SPG

    def return_spectra_graph_stats(self):
        if self.SPG is None:
            self.construct_spectra_graph()
        print("Stats for SPECTRA property graph (SPG)")
        print(f"Number of nodes: {self.SPG.number_of_nodes()}")
        print(f"Number of edges: {self.SPG.number_of_edges()}")
        num_connected_components = nx.number_connected_components(self.SPG)
        print(f"Number of connected components: {num_connected_components}\n\n")
        if num_connected_components > 1:
            print("Connected component stats")
            components = list(connected_components(self.SPG))
            densities = []
            for i, component in enumerate(components):
                subgraph = self.SPG.subgraph(component)
                print(f"Component {i} has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
                print(f"Density of component {i}: {nx.density(subgraph)}")
                densities.append(nx.density(subgraph))
                if is_clique(subgraph):
                    print(f"Component {i} is fully connected, all samples are similar to each other")
                else:
                    print(f"Component {i} is not fully connected")

            print(f"Average density {np.mean(densities)}")
    
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
        return [self.dataset[int(i)] for i in nodes]

    def generate_spectra_split(self, 
                               spectral_parameter, 
                               random_seed, 
                               test_size = 0.2):
        
        spectral_property_graph = self.SPG
        print(f"Generating SPECTRA split for spectral parameter {spectral_parameter} and dataset {self.dataset.name}")
        result = run_independent_set(spectral_parameter, spectral_property_graph, 
                                     seed = random_seed, 
                                     distribution = self.spectra_properties_loaded[2], 
                                     binary = self.binary)
        if len(result) <= 5:
            return None, None, None
        print(f"Number of samples in independent set: {len(result)}")
        train, test = self.spectra_train_test_split(result, test_size=test_size, random_state=random_seed)
        print(f"Train size: {len(train)}\tTest size: {len(test)}")
        cross_split_overlap = self.cross_split_overlap(self.get_samples(train), self.get_samples(test))
        print(f"Cross split overlap: {cross_split_overlap}\n\n\n")
        stats = {'SPECTRA_parameter': spectral_parameter, 'train_size': len(train), 'test_size': len(test), 'cross_split_overlap': cross_split_overlap}
        return train, test, stats
    
    def generate_spectra_splits(self, 
                                spectral_parameters, 
                                number_repeats, 
                                random_seed, 
                                test_size = 0.2, 
                                force_reconstruct = False,
                                data_path = None):
        
        #Random seed is a list of random seeds for each number
        name = self.dataset.name
        self.construct_spectra_graph(force_reconstruct = force_reconstruct)
        if self.binary:
            if nx.density(self.SPG) >= 0.4:
                raise Exception("Density of SPG is greater than 0.4, SPECTRA will not work as your dataset is too similar to itself. Please check your dataset and SPECTRA properties.")

        if data_path is None:
            data_path = ""
            if not os.path.exists(f"{name}_SPECTRA_splits"):
                os.makedirs(f"{name}_SPECTRA_splits")
            if not os.path.exists(f"{name}_spectral_property_graphs"):
                os.makedirs(f"{name}_spectral_property_graphs")
        else:
            if not os.path.exists(f"{data_path}/{name}_SPECTRA_splits"):
                os.makedirs(f"{data_path}/{name}_SPECTRA_splits")
            if not os.path.exists(f"{data_path}/{name}_spectral_property_graphs"):
                os.makedirs(f"{data_path}/{name}_spectral_property_graphs")

        splits = []
        for spectral_parameter in spectral_parameters:
            for i in range(number_repeats):
                if os.path.exists(f"{data_path}/{name}_SPECTRA_splits/SP_{spectral_parameter}_{i}") and not force_reconstruct:
                    print(f"Folder SP_{spectral_parameter}_{i} already exists. Skipping")
                elif force_reconstruct or not os.path.exists(f"{data_path}/{name}_SPECTRA_splits/SP_{spectral_parameter}_{i}"):
                    train, test, stats = self.generate_spectra_split(float(spectral_parameter), random_seed[i], test_size)
                    if train is not None:
                        if not os.path.exists(f"{data_path}/{name}_SPECTRA_splits/SP_{spectral_parameter}_{i}"):
                            os.makedirs(f"{data_path}/{name}_SPECTRA_splits/SP_{spectral_parameter}_{i}")
                
                        pickle.dump(train, open(f"{data_path}/{name}_SPECTRA_splits/SP_{spectral_parameter}_{i}/train.pkl", "wb"))
                        pickle.dump(test, open(f"{data_path}/{name}_SPECTRA_splits/SP_{spectral_parameter}_{i}/test.pkl", "wb"))
                        pickle.dump(stats, open(f"{data_path}/{name}_SPECTRA_splits/SP_{spectral_parameter}_{i}/stats.pkl", "wb"))
                    else:
                        print(f"Split for SP_{spectral_parameter}_{i} could not be generated since independent set only has one sample")
                
        return splits
    
    def return_split_stats(self, spectral_parameter, number):
        split_folder = f"./{self.dataset.name}_SPECTRA_splits/SP_{spectral_parameter}_{number}"
        if not os.path.exists(split_folder):
            raise Exception(f"Split folder {split_folder} does not exist")
        else:
            return pickle.load(open(f"{split_folder}/stats.pkl", "rb"))
    
    def return_split_samples(self, spectral_parameter, number):
        split_folder = f"./{self.dataset.name}_SPECTRA_splits/SP_{spectral_parameter}_{number}"
        if not os.path.exists(split_folder):
            raise Exception(f"Split folder {split_folder} does not exist")
        else:
            train = pickle.load(open(f"{split_folder}/train.pkl", "rb"))
            test = pickle.load(open(f"{split_folder}/test.pkl", "rb"))
            return [self.dataset[int(i)] for i in train], [self.dataset[int(i)] for i in test]
    
    def return_all_split_stats(self):
        SP = []
        numbers = []
        train_size = []
        test_size = []
        cross_split_overlap = []

        for folder in os.listdir(f"{self.dataset.name}_SPECTRA_splits"):
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
    
    def pre_calculate_spectra_properties(self, filename, force_recalculate = False):
        if os.path.exists(f"{filename}_precalculated_spectra_properties") and not force_recalculate:
            print(f"File {filename}_precalculated_spectra_properties already exists, set force_recalculate to True to recalculate")
        else:
            similarity_file = open(f"{filename}_precalculated_spectra_properties", 'w')

            for i in tqdm(range(len(self.dataset))):
                for j in range(i+1, len(self.dataset)):
                    similarity_file.write(f"{i}\t{j}\t{self.spectra_properties(self.dataset[i], self.dataset[j])}\n")
            
            similarity_file.close()
        self.load_spectra_precalculated_spectra_properties(filename)
    
    def load_spectra_precalculated_spectra_properties(self, filename):
        if not os.path.exists(f"{filename}_precalculated_spectra_properties"):
            raise Exception(f"File {filename}_precalculated_spectra_properties does not exist")
        else:
            self.spectra_properties_loaded = pd.read_csv(f"{filename}_precalculated_spectra_properties", sep = '\t', header = None)

        self.non_lookup_spectra_property = self.spectra_properties

        def lookup_spectra_property(x, y):
            if not is_integer(x) or not is_integer(y):
                return self.non_lookup_spectra_property(x, y)
            else:
                res1 = self.spectra_properties_loaded[(self.spectra_properties_loaded[0] == x) & (self.spectra_properties_loaded[1] == y)]
                res2 = self.spectra_properties_loaded[(self.spectra_properties_loaded[0] == y) & (self.spectra_properties_loaded[1] == x)]
                if len(res1) > 0:
                    return res1[2].values[0]
                elif len(res2) > 0:
                    return res2[2].values[0]
                else:
                    raise Exception(f"SPECTRA property between {x} and {y} not found in precalculated file")

        self.spectra_properties = lookup_spectra_property





    

