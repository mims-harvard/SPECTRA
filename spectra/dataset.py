from abc import ABC, abstractmethod

class SpectraDataset(ABC):

    def __init__(self, input_file, name):
        self.input_file = input_file
        self.name = name
        self.samples = self.parse(input_file)

    @abstractmethod
    def sample_to_index(self, idx):
        """
        Given a sample, return the data idx
        """
        pass
        
    
    @abstractmethod
    def parse(self, input_file):
        """
        Given a dataset file, parse the dataset file. 
        Make sure there are only unique entries!
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Return the length of the dataset
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        """
        Given a dataset idx, return the element at that index
        """
        pass