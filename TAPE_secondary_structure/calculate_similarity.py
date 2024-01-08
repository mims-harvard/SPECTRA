import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Bio import Align
from Bio.Align import substitution_matrices
import numpy as np
from tqdm import tqdm
import pickle
import json

def og_sequence_similarity(aligner, seq1, seq2):

    input = aligner.align(seq1, seq2)[0]

    count = 0
    diff = 0

    i = input[0]
    j = input[1]

    for orig, n in zip(i, j):
        if orig != n:
            diff += 1
        count += 1

    return 1 - diff/count

aligner = Align.PairwiseAligner()

aligner.match_score = 1.0
aligner.mismatch_score = -2.0
aligner.gap_score = -2.5

def create_sim_mat(calculated_values, ijs):
    print(f"Have to do {len(ijs)} values")
    for i,j in ijs:
        seqi=df[i]
        seqj=df[j]
        if ':' not in seqi and ':' not in seqj:
            identity=og_sequence_similarity(aligner, seqi,seqj)
            calculated_values[(i, j)] = identity
        
def extract_data(filenames):
    id_to_data = {}
    for filename in filenames:
        f = open(filename)
        data = json.load(f)

        for entry in data:
            id_to_data[entry['id']] = {}
            id_to_data[entry['id']]['primary'] = entry['primary']
    return id_to_data


train_to_data = extract_data(['secondary_structure/secondary_structure_train.json'])
val_to_data = extract_data(['secondary_structure/secondary_structure_valid.json'])
casp12_to_data = extract_data(['secondary_structure/secondary_structure_casp12.json'])
cb513_to_data = extract_data(['secondary_structure/secondary_structure_cb513.json'])
ts115_to_data = extract_data(['secondary_structure/secondary_structure_ts115.json'])

train_sequences = [train_to_data[i]['primary'] for i in train_to_data]
val_sequences = [val_to_data[i]['primary'] for i in val_to_data]
casp12_sequences = [casp12_to_data[i]['primary'] for i in casp12_to_data]
cb513_sequences = [cb513_to_data[i]['primary'] for i in cb513_to_data]
ts115_sequences = [ts115_to_data[i]['primary'] for i in ts115_to_data]

df = train_sequences+val_sequences+casp12_sequences+cb513_sequences+ts115_sequences
n=len(df)
print(f"Starting with {n} values")
import os 
found_chunks = False

for i in os.listdir('.'):
    if 'chunked_comparisons_30' in i:
        found_chunks = True
        chunked_comparisons = pickle.load(open('chunked_comparisons_30', 'rb'))

if not found_chunks:
    all_comparisons = [(i,j) for i in range(n) for j in range(i + 1, n)]
    print(f"Have to do {len(all_comparisons)} comparisons")
    chunked_comparisons = np.array_split(all_comparisons, 30)
    pickle.dump(chunked_comparisons, open('chunked_comparisons_30', 'wb'))

print(f"Chunked to {len(chunked_comparisons)} comparisons each {[len(i) for i in chunked_comparisons]}")
from multiprocessing import Process, Manager
manager = Manager()
d = manager.dict()
job = [Process(target=create_sim_mat, args=(d, i)) for i in chunked_comparisons]
_ = [p.start() for p in job]
_ = [p.join() for p in job]

import pickle
pickle.dump(dict(d), open('protein_similarity','wb') )
