import torch
from torch.utils.data import Sampler
import numpy as np
from sklearn.model_selection import train_test_split
from utils.general_utility_functions import *
from utils.constants import *
from tqdm import tqdm

def breakdown_file_prob(file_path, stratify = False):
	if stratify:
		probable_susceptible = []
		probable_resistant = []
		unknown = []
	else:
		strains = []

	for line in open(file_path, 'r').readlines():
		data = line.split('\t')
		strain = data[0]
		label_data = data[1].split(',')
		print(line)
		label = [float(label_data[0]), float(label_data[1])]
		
		if stratify:
			if label[0] >= 0.9:
				probable_susceptible.append(strain)
			elif label[1] >= 0.9:
				probable_resistant.append(strain)
			else:
				unknown.append(strain)	

		else:
			strains.append(strain)
	if stratify:
		return probable_susceptible, probable_resistant, unknown
	else:
		return strains

def breakdown_file(file_path, stratify = False):
	if stratify:
		pos_strains = []
		neg_strains = []
	else:
		strains = []

	for line in open(file_path, 'r').readlines():
		data = line.split('\t')
		strain = data[0]
		label = float(data[1])

		if strain != 'None':
			if stratify:
				if label:
					pos_strains.append(strain)
				else:
					neg_strains.append(strain)
			else:
				strains.append(strain)

	if stratify:
		return pos_strains, neg_strains
	else:
		np.random.shuffle(strains)
		return strains
class CovidSampler():
	def __init__(self, train_phenotypes):
		self.train_phenotypes = train_phenotypes
		self.below, self.above = self.threshold_samples()
		
		self.mean_score = False
		if self.mean_score:
			self.id_to_mean_barcode_score()


	def id_to_mean_barcode_score(self):
		id_to_score = {}

		num_to_score = {}
		for line in open('/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/Covid/data/covid_categorical_phenotypes','r').readlines():
			data = line.split('\t')
			num_to_score[data[0]] = float(data[1].rstrip())
		
		num_to_barcode = {}

		for line in open('/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/Covid/data/covid_barcodes','r').readlines():
			data = line.split('\t')
			num_to_barcode[data[0]] = data[1].rstrip()
			barcode = data[1].rstrip()
			if barcode not in barcode_to_score:
				barcode_to_score[barcode] = []
			barcode_to_score[barcode].append(num_to_score[data[0]])

		for barcode in barcode_to_score:
			barcode_to_score[barcode] = np.mean(barcode_to_score[barcode])

		num_to_score = {}
		for num in num_to_barcode:
			num_to_score[num] = barcode_to_score[num_to_barcode[num]]

		self.num_to_score = num_to_score



	def threshold_samples(self):
		above = []
		below = []

		for line in open(self.train_phenotypes, 'r').readlines():
			data = line.split('\t')

			if float(data[1]) < 0.2:
				below.append(data[0])
			else:
				above.append(data[0])
		
		return below, above

	def __len__(self):
		return len(self.below) + len(self.above)


	def __iter__(self):
		above_counter = 0
		below_counter = 0
		above = False
		while True:
			if not above:
				#yield np.random.choice(self.below, 1)[0]
				if self.mean_score:
					yield self.num_to_score[self.below[below_counter%len(self.below)]]
				else:
					yield self.below[below_counter%len(self.below)]
				below_counter += 1
				above = True

			else:
				#yield np.random.choice(self.above, 1)[0]
				if self.mean_score:
					yield self.num_to_score[self.above[above_counter%len(self.above)]]
				else:
					yield self.above[above_counter%len(self.above)]
				above_counter += 1
				above = False			


class BalancedBarcode():
	def __init__(self, drug, filename):
		if drug == 'covid':
			data_location = COVID_BARCODES
		else:
			data_location = "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/"
			mutational_barcodes = data_location + f"mutational_barcodes_reg_{drug}"
		self.breakdown_barcodes(mutational_barcodes)
		self.filename = filename


	def breakdown_barcodes(self, mutational_barcode):
		self.strain_to_barcode = {}

		for line in open(mutational_barcode, 'r').readlines():
			data = line.split('\t')
			self.strain_to_barcode[data[0]] = data[1].rstrip()

	def breakdown_train_file(self):
		strain_to_phenotype = {}
		barcode_to_phenotype = {}

		negative_barcode = []
		positive_barcode = []

		for line in open(self.filename):
			data = line.split('\t')
			strain_to_phenotype[data[0]] = data[1].rstrip()
			barcode = self.strain_to_barcode[data[0]]
			if barcode not in barcode_to_phenotype:
				barcode_to_phenotype[barcode] = {0:[],1:[]}

			if int(data[1].rstrip()):
				positive_barcode.append(barcode)
			else:
				negative_barcode.append(barcode)

			barcode_to_phenotype[barcode][int(data[1].rstrip())].append(data[0])

		return strain_to_phenotype, barcode_to_phenotype, negative_barcode, positive_barcode
	

	def __iter__(self):
		strain_to_phenotype, barcode_to_phenotype, negative_barcode, positive_barcode = self.breakdown_train_file()
		positive_barcode_counter = 0
		negative_barcode_counter = 0
		pos = True

		while True:
			if pos:
				barcode = positive_barcode[positive_barcode_counter%len(positive_barcode)]
				yield np.random.choice(barcode_to_phenotype[barcode][1])
				pos = False
				positive_barcode_counter += 1
			else:
				barcode = negative_barcode[negative_barcode_counter%len(negative_barcode)]
				yield np.random.choice(barcode_to_phenotype[barcode][0])
				pos = True
				negative_barcode_counter += 1


def generate_batches(lst, n):
	return [lst[i:i + n] for i in range(0, len(lst), n)]

class TestSampler(Sampler[int]):
	def __init__(self, train_file_path, test_file_path, type_sample = 'val', drug = 'PZA', balance = False, bin_sampler = False):
		self.train_strains = breakdown_file(train_file_path, False)
		self.test_strains  = breakdown_file(test_file_path, False)
		self.count = 0
		self.type_sample = type_sample
		self.balance = balance

		if self.balance:
			if drug == 'covid':
				self.barcode_mapping = get_all_barcode_mapping(COVID_BARCODES)
			elif drug == 'gfp':
				self.barcode_mapping = get_all_barcode_mapping(GFP_BARCODES)
			else:
				self.barcode_mapping = get_all_barcode_mapping(f"/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/mutational_barcodes_reg_{drug}")
			self.barcode_to_isolates = {}
			barcodes_present = [self.barcode_mapping[i] for i in self.test_strains]
			self.barcodes_present = list(set(barcodes_present))	
			for i in self.barcodes_present:
				self.barcode_to_isolates[i] = []
			[self.barcode_to_isolates[self.barcode_mapping[i]].append(i) for i in self.test_strains]

	def return_batches(self, batch_size, mode = 'test'):
		if mode == 'test':
			return generate_batches(self.test_strains, batch_size)
		else:
			return generate_batches(self.train_strains, batch_size)

	def __len__(self):
		if not self.balance:
			return len(self.test_strains)
		else:
			return len(self.barcodes_present)

	def __iter__(self):
		if not self.balance:
			if self.type_sample == 'val':
				while True:
					yield self.test_strains[self.count%len(self.test_strains)]
					self.count += 1
			else:
				while self.count < len(self.test_strains):
					yield self.test_strains[self.count]
					self.count += 1
		else:
			while True:
				yield random.sample(self.barcode_to_isolates[self.barcodes_present[self.count%len(self.barcodes_present)]], 1)[0]
				self.count += 1

class ProbSampler(Sampler[int]):
	def __init__(self, train_file_path, test_file_path):
		self.train_strains = breakdown_file_prob(train_file_path) 
		self.test_strains = breakdown_file_prob(test_file_path)

	def __len__(self):
		return len(self.train_strains)

	
	def get_val_sample(self, val_sample_size):
		return np.random.choice(self.test_strains, val_sample_size)

	def __iter__(self):
		while True:
			strain = np.random.choice(self.train_strains, 1)[0]
			yield strain

class ProbSampler_Stratified(Sampler[int]):
	def __init__(self, train_file_path, test_file_path):
		self.train_ps, self.train_pr, self.train_unknown = breakdown_file_prob(train_file_path, stratify = True) 
		self.test_ps, self.test_pr, self.test_unknown = breakdown_file_prob(test_file_path, stratify = True)

	def __len__(self):
		return len(self.train_ps) + len(self.train_pr) + len(self.train_unknown)

	def get_sample(self, mode, num_stratify):
		if mode == 'train':
			true_ps, true_pr, true_unknown = self.train_ps, self.train_pr, self.train_unknown
		elif mode == 'test':
			true_ps, true_pr, true_unknown = self.test_ps, self.test_pr, self.test_unknown

		if num_stratify == 0:
			return np.random.choice(true_ps, 1)[0]
		elif num_stratify == 1:
			return np.random.choice(true_pr, 1)[0]
		elif num_stratify == 2:
			return np.random.choice(true_unknown, 1)[0]


	def get_val_sample(self, val_sample_size):
		strains_to_test = []
			
		num_stratify = 0
		for i in range(val_sample_size):
			strains_to_test.append(self.get_sample('test', num_stratify))
			num_stratify = (num_stratify+1)%3

		return strains_to_test

	def __iter__(self):	
		num_stratify = 0
		while True:
			yield self.get_sample('train', num_stratify)
			num_stratify = (num_stratify + 1)%3

class SimpleSamplerCategorical(Sampler[int]):
	"""
	Simple sampler for categorical data

	"""

	def __init__(self, train_file_path, test_file_path, drug = 'PZA', balance = False, bin_sampler = False):
		self.train_strains, self.train_to_value, self.train_values = self.breakdown_file(train_file_path) 
		self.test_strains, self.test_to_value, self.test_values = self.breakdown_file(test_file_path)
		self.count = 0
		self.bin_sampler = bin_sampler
		if self.bin_sampler:
			self.bin_to_sample = self.generate_bins()

		if drug == 'covid':
			import random
			random.shuffle(self.train_strains)		

	def breakdown_file(self, file_path):
		samples = []
		sample_to_value = {}
		values = []	
		for line in open(file_path, 'r').readlines():
			data = line.split('\t')
			samples.append(data[0])
			sample_to_value[data[0]] = float(data[1])
			values.append(float(data[1]))

		return samples, sample_to_value, values

	def generate_bins(self):
		#import pdb; pdb.set_trace()
		number_bins = self.bin_sampler
		increments = (max(self.train_values) - min(self.train_values))/number_bins
		bin_to_samples = {}
		for i in range(number_bins):
			bin_to_samples[i] = []
		minimum_value = min(self.train_values)		
		for sample in self.train_to_value:	
			value = self.train_to_value[sample]
			bin_number = int((value - minimum_value)/increments)
			if bin_number == self.bin_sampler:
				bin_number = self.bin_sampler - 1
			bin_to_samples[bin_number].append(sample)
			#import pdb; pdb.set_trace()

	
		for num in bin_to_samples:
			bin_to_samples[num] = np.array(bin_to_samples[num])
		

		return bin_to_samples


	def __len__(self):
		return len(self.train_strains)

	def __iter__(self):
		while True:
			if not self.bin_sampler:
				yield self.train_strains[self.count%len(self.train_strains)]
				self.count += 1
			else:
				#import pdb; pdb.set_trace()
				yield np.random.choice(self.bin_to_sample[self.count%self.bin_sampler], 1)[0]
				self.count += 1	
	

class SimpleSampler(Sampler[int]):
	"""	
	Splits strains to train and val, we train on train, test on val

	"""
	
	def __init__(self, train_file_path, test_file_path, drug = 'PZA', balance = False):
		self.pos_train_strains, self.neg_train_strains = self.breakdown_file(train_file_path)
		self.pos_test_strains, self.neg_test_strains = self.breakdown_file(test_file_path)
		if drug == 'covid':
			self.barcode_mapping = get_all_barcode_mapping(COVID_BARCODES)
		else:
			self.barcode_mapping = get_all_barcode_mapping(f"/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/mutational_barcodes_reg_{drug}")
		self.balance = balance
		np.random.seed(10)	
		self.pos_train_strains = np.array(self.pos_train_strains)
		self.neg_train_strains = np.array(self.neg_train_strains)			
	
	def breakdown_file(self, file_path):
		pos_strains = []
		neg_strains = []

		for line in open(file_path, 'r').readlines():
			data = line.split('\t')
			strain = data[0]
			label = int(data[1])

			if strain != 'None':	
				if label:
					pos_strains.append(strain)
				else:
					neg_strains.append(strain)
			#strains.append(line.split('\t')[0])

		return pos_strains, neg_strains

	def __len__(self):
		#return 1000
		return len(self.pos_train_strains) + len(self.neg_train_strains)
	
	def get_val_sample(self, val_sample_size):
		strains_to_test = []
		#So we sample val sample strains each with 2 genes, so test 200 random val genes!
		pos = True
		
		for i in range(val_sample_size):
			if pos:
				strains_to_test.append(np.random.choice(self.pos_test_strains, 1)[0])
				pos = False
			else:
				strains_to_test.append(np.random.choice(self.neg_test_strains, 1)[0])
				pos = True			

		return strains_to_test

	def get_train_sample_num(self, train_sample_size):
		strains_to_train = []
		#So we sample val sample strains each with 2 genes, so test 200 random val genes!
		pos = True

		for i in range(train_sample_size):
			if pos:
				strains_to_train.append(np.random.choice(self.pos_train_strains, 1)[0])
				pos = False
			else:
				strains_to_train.append(np.random.choice(self.neg_train_strains, 1)[0])
				pos = True

		return strains_to_train

	def get_alternate_strain(self, strain, split_strain):
		
		if strain in self.pos_train_strains:
			return np.random.choice(self.pos_train_strains, 1)[0]	
		elif strain in self.neg_train_strains:
			return np.random.choice(self.neg_train_strains, 1)[0]
		elif strain in self.pos_test_strains:
			return np.random.choice(self.pos_test_strains, 1)[0]
		elif strain in self.neg_test_strains:
			return np.random.choice(self.neg_test_strains, 1)[0]
		else:
			raise Exception(f"{strain} was not found anywhere")

	def get_train_sample(self):
		strain = np.random.choice(self.train_strains, 1)[0]
	
		return strain
	
	def __iter__(self):
		pos = True
		while True:
			if pos:
				strain = np.random.choice(self.pos_train_strains, 1)[0]
				#while not self.barcode_mapping[strain]:
					#print("Here we go remapping")
				#	strain = np.random.choice(self.pos_train_strains, 1)[0] 
				pos = False
			else:
				strain = np.random.choice(self.neg_train_strains, 1)[0]
				#while not self.barcode_mapping[strain]:
					#print("HEre we go remapping")
				#	strain = np.random.choice(self.neg_train_strains, 1)[0]
				pos = True
			#yield "SAMEA1485226"
			yield strain


class NaiveSampler(Sampler[int]):
	"""	
	Just always return one strain see if we can overfit to that one strain

	"""
	
	def __init__(self, train_file_path, test_file_path):
		self.train_strains = self.breakdown_file(train_file_path)
		self.test_strains = self.breakdown_file(test_file_path)
		self.strain_to_return = np.random.choice(self.train_strains, 1)[0]
		
	def breakdown_file(self, file_path):
		strains = []
		for line in open(file_path, 'r').readlines():
			strains.append(line.split('\t')[0])
		return strains

	def __len__(self):
		return len(self.train_strains)
	
	def get_val_sample(self, val_sample_size):
		strains_to_test = []
		#So we sample val sample strains each with 2 genes, so test 200 random val genes!
		for i in list(range(val_sample_size)):
			strains_to_test.append(self.strain_to_return)

		return strains_to_test

	def get_train_sample_num(self, train_sample_size):
		strains_to_train = []
		#So we sample val sample strains each with 2 genes, so test 200 random val genes!
		for i in list(range(len(self.train_strains))):
			strains_to_train.append(self.strain_to_return)
		return strains_to_train

	def get_train_sample(self):
		return self.strain_to_return
	
	def __iter__(self):
		while True:
			yield self.strain_to_return
