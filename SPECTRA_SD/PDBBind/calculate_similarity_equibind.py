import pandas as pd
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from Bio import Align
from Bio.Align import substitution_matrices
import numpy as np
import pickle
import networkx as nx
from equibind_similarity_util import calc_similarity

G = nx.Graph()

def create_sim_mat(ijs, calculated_values):
    print(f"Have to do {len(ijs)} values")

    for i,j in ijs:
        seq_1=df.iloc[i].seq
        seq_2=df.iloc[j].seq

        smile_1=df.iloc[i].smiles
        smile_2=df.iloc[j].smiles

        mol_similarity, seq_similarity = calc_similarity(seq_1, seq_2, smile_1, smile_2)
        if mol_similarity == -1:
            if seq_similarity > 0.3:
                calculated_values.append((i,j))
        elif mol_similarity > 0.99 or seq_similarity > 0.3:
                calculated_values.append((i,j))
        
df = pd.read_csv('equibind_dataset.csv')
n=len(df)
print(f"Starting with {n} values")
import os 
found_chunks = False

for i in os.listdir('.'):
    if 'chunked_comparisons_30_equibind' in i:
        found_chunks = True
        chunked_comparisons = pickle.load(open('chunked_comparisons_30_equibind', 'rb'))

if not found_chunks:
    all_comparisons = [(i,j) for i in range(n) for j in range(i + 1, n)]
    print(f"Have to do {len(all_comparisons)} comparisons")
    chunked_comparisons = np.array_split(all_comparisons, 30)
    pickle.dump(chunked_comparisons, open('chunked_comparisons_30_equibind', 'wb'))

print(f"Chunked to {len(chunked_comparisons)} comparisons each {[len(i) for i in chunked_comparisons]}")

from multiprocessing import Process, Manager
manager = Manager()

d = manager.list()
job = [Process(target=create_sim_mat, args=(i, d)) for i in chunked_comparisons]
_ = [p.start() for p in job]
_ = [p.join() for p in job]

import pickle
pickle.dump(list(d), open('equibind_edges','wb') )
