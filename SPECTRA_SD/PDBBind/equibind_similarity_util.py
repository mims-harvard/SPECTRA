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

def og_sequence_similarity(aligner, seq1, seq2):
    #Function to calculate sequence similarity between two sequences
    #Aligns sequence then iterates over alignment and counts number of aligned positions
    #Returns proportion of aligned positions as sequence similarity

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

def calc_similarity(seq_1, seq_2, mol_1, mol_2):
    #Returns sequence similarity value and molecular similarity (if RDKit accepts the molecular string)
    mol_1 = capital_within_bracket(mol_1)
    mol_1 = mol_1.replace('L', '').upper().replace('L', '')
    if 'OH2' in mol_1:
        mol_1 = mol_1.replace('OH2', 'OH1')
    if 'BR' in mol_1:
        mol_1 = mol_1.replace('BR', 'Br')
    if mol_1 == 'C([C@H]1[C@@H]([C@@H](C[NH2+]1)O)O)[N]1=NN=C(C2CCC(CC2)OCCOCCNN=N)C1':
        mol_1 = 'C([C@H]1[C@@H]([C@@H](C[NH2+]1)O)O)[N+]1=NN=C(C2CCC(CC2)OCCOCCNN=N)C1'
    
    mol_2 = capital_within_bracket(mol_2)
    mol_2 = mol_2.replace('L', '').upper().replace('L', '')
    if 'OH2' in mol_2:
        mol_2 = mol_2.replace('OH2', 'OH1')
    if 'BR' in mol_2:
        mol_2 = mol_2.replace('BR', 'Br')
    if mol_2 == 'C([C@H]1[C@@H]([C@@H](C[NH2+]1)O)O)[N]1=NN=C(C2CCC(CC2)OCCOCCNN=N)C1':
        mol_2 = 'C([C@H]1[C@@H]([C@@H](C[NH2+]1)O)O)[N+]1=NN=C(C2CCC(CC2)OCCOCCNN=N)C1'

    if ':' in seq_1:
        seq_1 = seq_1.split(':')
    else:
        seq_1 = [seq_1]

    if ':' in seq_2:
        seq_2 = seq_2.split(':')
    else:
        seq_2 = [seq_2]

    try:
        mol_similarity = return_molecule_molecule_similarity(mol_1, mol_2)
    except Exception:
        #print(f"ERROR {mol_1} and {mol_2} FAILED")
        mol_similarity = -1
   
    avg_seq_similarity = []
    for i in seq_1:
        for j in seq_2:
            avg_seq_similarity.append(og_sequence_similarity(aligner, i,j))
    
    return mol_similarity, np.mean(avg_seq_similarity)

aligner = Align.PairwiseAligner()

aligner.match_score = 1.0
aligner.mismatch_score = -2.0
aligner.gap_score = -2.5

def determine_molecular_similarity(i, j):
    #Calculate molecular similarity, code borrowed from LPPDBBind:
    #https://github.com/THGLab/LP-PDBBind/blob/master/dataset_creation/calc_similarities.ipynb

    fp_1=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i),2,nBits=1024)
    fp_2=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(j),2,nBits=1024)
    sim=DataStructs.DiceSimilarity(fp_1, fp_2)
    if sim == 1:
        fp_1=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i),4,nBits=1024)
        fp_2=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(j),4,nBits=1024)
        sim=DataStructs.DiceSimilarity(fp_1, fp_2)
        if sim == 1:
            fp_1=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i),10,nBits=1024)
            fp_2=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(j),10,nBits=1024)
            sim=DataStructs.DiceSimilarity(fp_1, fp_2)
            if sim==1:
                if (i != j):
                    sim=0.99
    return sim


def return_molecule_molecule_similarity(mol_1, mol_2):
    #Convert to canoncial smile format before calculating similarity
    mol_1_result = Chem.CanonSmiles(mol_1)
    mol_2_result = Chem.CanonSmiles(mol_2)
    return determine_molecular_similarity(mol_1_result, mol_2_result)

def capital_within_bracket(mol):
    #Some heuristics to try to get molecule in correct format for RDKit
    rebuild_mol = ''
    cap = False
    
    for i in mol:
        if i == '[' or i == '(':
            cap = True
        elif i == ']' or i == ')':
            cap = False
        
        if cap or i == 'p':
            rebuild_mol = rebuild_mol + i.upper()
        else:
            rebuild_mol = rebuild_mol + i
    return rebuild_mol

def is_similar(seq_1, seq_2, smile_1, smile_2):
        #Returns if similar or not based on sequence and molecular similarity
        mol_similarity, seq_similarity = calc_similarity(seq_1, seq_2, smile_1, smile_2)
        if mol_similarity == -1:
            if seq_similarity > 0.3:
                return True
        elif mol_similarity > 0.99 or seq_similarity > 0.3:
            return True
        return False
