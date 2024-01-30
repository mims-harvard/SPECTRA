import numpy as np
import random
import pandas as pd
from utils.constants import *

"""
Resistance mapping function
"""

def PYRAZINAMIDE(row):
    if row['PYRAZINAMIDE'] == 'R':
        return 1
    elif row['PYRAZINAMIDE'] == 'S':
        return 0
    else:
        return -1

def RIFAMYCIN(row):
    if(row['RIFABUTIN'] == 'R' or row['RIFAMPICIN'] == 'R'):
        return 1
    elif(row['RIFABUTIN'] == 'S' or row['RIFAMPICIN'] == 'S'):
        return 0
    else:
        return -1

class Resistance_Information():
    
    def __init__(self, drug):
        self.df = pd.read_csv(RESISTANCE_METADATA, sep = ',')
        self.df['PZA'] = self.df.apply(lambda row: PYRAZINAMIDE(row), axis=1)
        self.df['RIF'] = self.df.apply(lambda row: RIFAMYCIN(row), axis=1)
        self.df.index = self.df['Isolate']
        self.isolates_in_resistance = list(self.df['Isolate'].values)
        self.drug = drug
        self.generate_name_mapping()    
    
    def generate_name_mapping(self):
        name_mapping = {}
        
        for line in open(NAME_MAPPING, 'r').readlines()[1:]:
            data = line.split(',')
            name_mapping[data[0]] = [data[1], data[2].rstrip()]
        
        self.name_mapping = name_mapping

    def return_total_number(self):
        num_res = 0
        num_sus = 0

        for isolate in self.isolates_in_resistance:   
            if self.df.at[isolate, self.drug] == 0:
                num_sus += 1
            elif self.df.at[isolate, self.drug] == 1:
                num_res += 1

        return num_res, num_sus
    
    def __getitem__(self, i):
        i = i.replace('-', '')
        
        real_name = False
        
        if i in self.isolates_in_resistance:
            real_name = i
        else:
            for pot_name in self.name_mapping[i]:
                if pot_name in self.isolates_in_resistance:
                    real_name = pot_name
                    break
                    
        if not real_name:
            raise Exception("Could not find name")
        
        return self.df.at[real_name, self.drug]

def get_resistance_breakdown(samples, drug):
    drug_info = Resistance_Information(drug)
    num_resistant = 0
    num_susceptible = 0

    for sample in samples:
        if drug_info[sample] == 0:
            num_susceptible += 1
        elif drug_info[sample] == 1:
            num_resistant += 1
        else:
            print(f"Could not find info {sample}")

    print(f"Number Resistant {num_resistant}")
    print(f"Number Susceptible {num_susceptible}")
    return num_resistant, num_susceptible

"""
BARCODE UTILITY FUNCTION
dealing with barcode operations
mapping strain to barcode

"""

region_information_file = "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/Region_Data"

def get_region_mapping():
    region_mapping = {}
    for line in open(region_information_file, 'r').readlines():
        if 'name' not in line:
            data = line.split('\t')
            region_mapping[f'[{data[2]}, {data[3]}]'] = data[1]
            
    return region_mapping

region_information = get_region_mapping()

def remap_barcode(barcode, mapping = True):
    if not barcode or not mapping:
        return barcode

    new_barcode = ""
    for mut in barcode.split('-'):
        new_mut = ""
        new_mut = new_mut + region_information[mut.split(':')[0]].replace('-','_') + ':'
        new_mut = new_mut + ':'.join(mut.split(':')[1:])
        new_barcode = new_barcode + '-' + new_mut
    
    return new_barcode[1:]

def return_number_isolates(list_barcodes, barcode_file, mapping = True):
    number = 0
    
    for line in open(barcode_file, 'r').readlines():
        data = line.split('\t')
        if data[1].rstrip():
            barcode = remap_barcode(data[1].rstrip(), mapping)
            if barcode in list_barcodes:
                number += 1
        else:
            if 'NO_MUTATION' in list_barcodes:
                number += 1
    
    return number

def get_all_barcode_mapping(barcode_file, mutations = False):
	barcode_mapping = {}
	for line in open(barcode_file, 'r').readlines():
		data = line.split('\t')
		if mutations:
			barcode_mapping[data[0]] = data[1].rstrip().split('-')
		else:
			barcode_mapping[data[0]] = data[1].rstrip()
	return barcode_mapping

def get_strains_barcode_file(barcode_file):
	strains = []
	for line in open(barcode_file, 'r').readlines():
		data = line.split('\t')
		strains.append(data[0])
	return strains

def return_barcode_isolates(list_isolates, barcode_file):
    #Function that takes in a list of isolates and returns their corresponding barcodes
    barcodes_identified = []

    for line in open(barcode_file, 'r').readlines():
        data = line.split('\t')
        if data[0].rstrip() in list_isolates:
           barcodes_identified.append(data[1].rstrip())

    return barcodes_identified

def return_all_barcodes(barcode_file):
    #Return all barcodes in barcode file
    barcodes_seen = []

    for line in open(barcode_file, 'r').readlines():
        data = line.split('\t')
        if data[1].rstrip():
            barcodes_seen.append(data[1].rstrip())

    return list(set(barcodes_seen))

def return_isolates_mutation(mutation, barcode_file):
    #Function that takes in a  mutation and returns isolates with that mutation
    
    isolates_identified = []
    
    for line in open(barcode_file, 'r').readlines():
        data = line.split('\t')
        if data[1].rstrip():
            barcode = remap_barcode(data[1].rstrip())
            if mutation in barcode:
                isolates_identified.append(data[0])
    
    return isolates_identified

def generate_isolates_barcode(barcode_file, mapping = True):
    #Create a dictionary that maps barcode to isolate
    mapping = {}
    
    for line in open(barcode_file, 'r').readlines():
        data = line.split('\t')
        if data[1].rstrip():
            barcode = data[1].rstrip()
            if barcode not in mapping:
                mapping[barcode] = []
            mapping[barcode].append(data[0].rstrip())
        else:
            if 'NO_MUTATION' not in mapping:
                mapping['NO_MUTATION'] = []
            mapping['NO_MUTATION'].append(data[0].rstrip())

    return mapping

def return_isolates_barcodes(barcode_to_isolates, list_barcode):
    isolates = []
    for i in list_barcode:
        isolates.extend(barcode_to_isolates[i])
    
    return isolates
           
def return_isolates_barcode(list_barcodes, barcode_file, mapping = True):
    #Function that takes in a list of barcodes and returns their correponding isolates
    return return_isolates_barcodes(generate_isolates_barcode(barcode_file, mapping), list_barcodes)

def return_all_mutations(list_barcodes, mapping = True):
    #Return all the mutations associated with a list of barcodes

    mutations = []

    for barcode in list_barcodes:
        for mut in remap_barcode(barcode, mapping).split('-'):
            mutations.append(mut)

    return list(set(mutations))

def identify_shared_mutations(barcode_file, mapping = True):
    mutation_pool = set()
    barcode_pool = set()
    
    for line in open(barcode_file, 'r').readlines():
        data = line.split('\t')
        if data[1].rstrip():
            barcode_pool.add(remap_barcode(data[1].rstrip(), mapping))
            for mut in remap_barcode(data[1].rstrip(), mapping).split('-'):
                mutation_pool.add(mut)

    mutation_counts = {}
    for mut in mutation_pool:
        mutation_counts[mut] = 0
    
    for barcode in barcode_pool:
        for mut in mutation_pool:
            if mut in barcode:
                mutation_counts[mut] += (1/len(barcode_pool))
    
    return mutation_counts


def return_mutational_scores(barcode_file, mapping = True):
    mutation_frequencies = identify_shared_mutations(barcode_file, mapping)
    
    barcode_to_mutational_score = {}
    
    for line in open(barcode_file, 'r').readlines():
        data = line.split('\t')
        if data[1].rstrip():
            barcode_to_interogate = remap_barcode(data[1].rstrip(), mapping)
            mutational_scores = []
            for mut in barcode_to_interogate.split('-'):
                mutational_scores.append(mutation_frequencies[mut])
            
            barcode_to_mutational_score[barcode_to_interogate] = np.mean(mutational_scores)
    
    
    return barcode_to_mutational_score


def get_filtered_mutations(barcode_file):
    drug_mut = identify_shared_mutations(barcode_file)
    print(f"Mean mutational frequency {np.mean(list(drug_mut.values()))}")
    print(f"STD mutational frequency {np.std(list(drug_mut.values()))}")
    print(f"3 STD Away {np.mean(list(drug_mut.values())) + 3*(np.std(list(drug_mut.values())))}")
    threshold = np.mean(list(drug_mut.values())) + 3*(np.std(list(drug_mut.values())))
    filtered_mutations = []
    for mut,val in drug_mut.items():
        if val >= threshold:
            filtered_mutations.append([mut, val])
    return filtered_mutations


def return_barcode_mapping(list_barcodes, barcode_file, mapping = True):
    barcode_mapping = {}
    
    for line in open(barcode_file, 'r').readlines():
        data = line.split('\t')
        if data[1].rstrip():
            barcode = remap_barcode(data[1].rstrip(), mapping)
            if barcode not in barcode_mapping:
                barcode_mapping[barcode] = 0
            barcode_mapping[barcode] += 1
        else:
            if 'NO_MUTATION' not in barcode_mapping:
                barcode_mapping['NO_MUTATION'] = 0
            barcode_mapping['NO_MUTATION'] += 1

    to_return_mapping = {}

    for i in list_barcodes:
        to_return_mapping[i] = barcode_mapping[i]

    return to_return_mapping

def return_barcode_mapping_pheno(list_barcodes, barcode_file, drug, mapping = True):
	res_info = Resistance_Information(drug)
	barcode_res_mapping = {}
	for i in list_barcodes:
		barcode_res_mapping[i] = {'R':0, 'S':0}
	
	for line in open(barcode_file, 'r').readlines():
		data = line.split('\t')
		if data[1].rstrip():
			barcode = remap_barcode(data[1].rstrip(), mapping = True)
			if barcode in list_barcodes:
				if res_info[data[0]] == 1.0:
					barcode_res_mapping[barcode]['R'] += 1
				elif res_info[data[0]] == 0.0:
					barcode_res_mapping[barcode]['S'] += 1
	return barcode_res_mapping		
	

"""
Function to deal with mutational data splits
"""


def breakdown_mutational_split(mutational_split):
    isolates_present = []

    for line in open(mutational_split, 'r').readlines():
        data = line.split('\t')
        isolates_present.append(data[0].rstrip())

    return isolates_present  


def determine_mutational_overlap(mutational_split_file_one, mutational_split_file_two, barcode_file):
    #For two mutational splits print all mutations that are overlapping

    barcodes_one = return_barcode_isolates(breakdown_mutational_split(mutational_split_file_one), barcode_file)
    barcodes_two = return_barcode_isolates(breakdown_mutational_split(mutational_split_file_two), barcode_file)

    mutations_one = return_all_mutations([i for i in barcodes_one if i])
    mutations_two = return_all_mutations([i for i in barcodes_two if i])
    #import pdb; pdb.set_trace()
    for mut_one in mutations_one:
        for mut_two in mutations_two:
            if mut_one == mut_two:
                print(f"{mut_one} and {mut_two} overlap")

"""
SUBSET SUM FUNCTION
clean up the mutational data splits

"""
   
def split_barcodes_subset_sum(independent_set, barcode_file, test_size = 0.2, seed = None, random_split = False):    
    #Can I find a split of barcodes that meets this sample number?
    #This is the subset sum problem we have a set of integers can we partition into a subset to reach a number
 
    if 'covid' in barcode_file or 'gfp' in barcode_file:
        mapping = False
    else:
        mapping = True
    
    if seed is not None:
        random.seed(seed)
    
    total_number_isolates = 0
    barcode_to_number = return_barcode_mapping(independent_set, barcode_file, mapping)
    
    for i in barcode_to_number:
        total_number_isolates += barcode_to_number[i]
    
    number_barcodes_train = len(independent_set)*(1-test_size)
    number_barcodes_test = len(independent_set)*test_size
    number_samples_train = total_number_isolates*(1-test_size)
    number_samples_test = total_number_isolates*test_size
    
    def rate_split(train_set, train_set_sum, test_set, test_set_sum):
        qual_train_barcodes = abs(len(train_set) - number_barcodes_train)
        qual_test_barcodes = abs(len(test_set) - number_barcodes_test)
        qual_train_samples = abs(train_set_sum - number_samples_train)
        qual_test_samples = abs(test_set_sum - number_samples_test)
        return qual_train_barcodes + qual_test_barcodes + qual_train_samples + qual_test_samples

    best_train_set = []
    best_train_set_sum = 0
    best_test_set = []
    best_test_set_sum = 0
    best_score = 100000000
    
    for iterations in range(500):
    
    
        train_set = []
        train_set_sum = 0
        test_set = []
        test_set_sum = 0

        #Randomly shuffle an independent set
        random.shuffle(independent_set)

        discarded_barcodes = []
        for barcode in independent_set:
            if random.random() < 1 - test_size:
                #Into the train you go
                #Only consider the barcode if it keeps the sum to less than the total
                if train_set_sum + barcode_to_number[barcode] < number_samples_train:
                    #If it is less than then we just add
                    train_set.append(barcode)
                    train_set_sum += barcode_to_number[barcode]
                else:
                    discarded_barcodes.append(barcode)
            else:
                #Into the test you go
                #Only consider the barcode if it keeps the sum to less than the total
                if test_set_sum + barcode_to_number[barcode] < number_samples_test:
                    #If it is less than we just add
                    test_set.append(barcode)
                    test_set_sum += barcode_to_number[barcode]
                else:
                    discarded_barcodes.append(barcode)

        for barcode in discarded_barcodes:
            if random.random() < 1 - test_size:
                train_set.append(barcode)
                train_set_sum += barcode_to_number[barcode]
            else:
                test_set.append(barcode)
                test_set_sum += barcode_to_number[barcode]
        
        score = rate_split(train_set, train_set_sum, test_set, test_set_sum)
        if score < best_score:
            best_score = score 
            best_train_set = train_set
            best_train_set_sum = train_set_sum
            best_test_set = test_set
            best_test_set_sum = test_set_sum
    
    print(f"Number of barcodes in train {len(best_train_set)} number of samples {best_train_set_sum}")
    print(f"Number of barcodes in test {len(best_test_set)} number of samples {best_test_set_sum}")
    print(f"Best Score {best_score}")
    print(f"Target number of barcodes in train {number_barcodes_train}")
    print(f"Target number of barcodes in test {number_barcodes_test}")
    print(f"Target number of samples in train {number_samples_train}")
    print(f"Target number of samples in test {number_samples_test}")
    
    return best_train_set, best_test_set



