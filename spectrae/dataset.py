from abc import ABC, abstractmethod
from typing import List, Dict

class SpectraDataset(ABC):

    def __init__(self, input_file, name):
        self.input_file = input_file
        self.name = name
        self.sample_to_index = self.parse(input_file)
        self.samples = list(self.sample_to_index.keys())
        self.samples.sort()
    
    @abstractmethod
    def parse(self, input_file: str) -> Dict:
        """
        Given a dataset file, parse the dataset file to return a dictionary mapping a sample ID to the data
        """
        raise NotImplementedError("Must implement parse method to use SpectraDataset, see documentation for more information")

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Given a dataset idx, return the element at that index
        """
        if isinstance(idx, int):
            return self.sample_to_index[self.samples[idx]]
        return self.sample_to_index[idx]
    
    def index(self, value):
        """
        Given a value, return the index of that value
        """
        if value not in self.samples:
            raise ValueError(f"{value} not in the dataset")
        return self.samples.index(value)