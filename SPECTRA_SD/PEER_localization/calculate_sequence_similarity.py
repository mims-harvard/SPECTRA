from torchdrug import datasets
from torchdrug.data.protein import Protein
from Bio import Align
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import random
import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap

def align_overlap(input):
	i = input[0]
	j = input[1]

	count = 0
	diff = 0

	for orig, n in zip(i, j):
		if orig != n:
			diff += 1
		count += 1

	return 1 - diff/count

def compare_to_all(i, compare, aligner):
	result = []
	for j in compare:
		if len(i) != len(j):
			input = aligner.align(i, j)[0]
		else:
			input = [i,j]
		result.append([i,j,align_overlap(input)])
	return result


def extract_sequence_sequence_similarity(chunk, sequences, test_file):

	aligner = Align.PairwiseAligner()

	aligner.match_score = 1.0
	aligner.mismatch_score = -2.0
	aligner.gap_score = -2.5
	
	import multiprocessing as mp
	print("Number of processors: ", mp.cpu_count())
	pool = mp.Pool(16)

	chunks_to_run = []
	count = 0
	for i in sequences:
		chunks_to_run.append((i, sequences[count:], aligner))
		count += 1
	
	count = 0
	for result in tqdm(pool.istarmap(compare_to_all, chunks_to_run), total=len(sequences)):
		for i,j,k in result:
			count += 1
			test_file.write(f"{i}\t{j}\t{k}\n")
	
	print(f"Final count {count}")

full_sequences = []
#filepath to fasta file with all PEER localization dataset sequences
filename = ""

for line in open(filename, 'r').readlines():
	if '>' not in line:
		full_sequences.append(line.rstrip())

test_file = open('localization_sequences_comparison','w')
full_sequences = list(set(full_sequences))
extract_sequence_sequence_similarity(full_sequences, full_sequences, test_file)
