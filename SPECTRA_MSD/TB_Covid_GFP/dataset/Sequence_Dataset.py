from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Sampler
import random
from utils.constants import *
from utils.generate_barcode import *
from utils.mod_alignment_utils import *
from tqdm import tqdm
from utils.check_mutational_splits import *
from utils.general_utility_functions import *
import esm
import numpy as np

class Sequence_Dataset(LightningDataModule):
	def __init__(self,
		alignment_location:		str = "/example/filepath",
		input_gff_file:         str = "/example/filepath",
		batch_size:             int = 64,
		eval_size:              int = 64,
		train_strains:          str = "/example/filepath",
		test_strains:           str = "/example/filepath",
		reference_nucleotide:   str = "/example/filepath",
		full_reference_sequence:str = "/example/filepath",
		sequence_generation:	bool = False,
		unirep_embedding: 	bool = False,
		esm_embedding:		bool = False,
		frozen:			bool = False,
		drug:					str = "DRUG",
		region:					str = "/example/filepath",
		random_mode:				bool = False,
		use_pregenerated:			bool = False,
		generate_pregenerated:			bool = False,
		model_type:				str = "esm",
		nucleotide:				bool = True,
                embedding_file:				str = None
		):

		self.batch_size = batch_size
		self.eval_size = eval_size

		self.sequence_generation = sequence_generation
		self.unirep_embedding = unirep_embedding	
		self.esm_embedding = esm_embedding
		self.frozen = frozen
		self.drug = drug
		self.model_type = model_type
		self.nucleotide = nucleotide
		self.embedding_file = embedding_file

		if self.sequence_generation:
			self.strain_to_sequence =  self.breakdown_sequence()
		else:
			self.fetcher = GenerateBarcode(drug, use_pregenerated)

		self.random_mode = random_mode

		self.strain_to_label = {}
		self.train_strains = self.breakdown_strains(train_strains)
		self.test_strains = self.breakdown_strains(test_strains)

		if self.drug == "covid":
			covid_min = -9.101593596174292
			covid_max = 0

			barcode_to_phenotype = {}
			for line in open(COVID_CATEGORICAL_PHENOTYPES, 'r').readlines():
				data = line.split('\t')
				barcode_to_phenotype[data[0]] = (np.log(float(data[1].rstrip())) - covid_min)/(covid_max - covid_min)

			for i in self.strain_to_label:
				self.strain_to_label[i] = barcode_to_phenotype[i]
		
		self.regions = self.breakdown_regions(region, drug)
		#This is the whole if have less than 10 regions throw out idea, not doing that now
		self.filter_data = False
		self.pre_defined_encoding_vector = None
		#Make this below to 0 to recalculate longest length
		if drug == 'PZA':
			self.longest_length = 3120
		elif drug == 'RIF':
			self.longest_length = 7526
		elif drug == 'INH':
			self.longest_length = 4000
		elif drug == 'gfp':
			if self.model_type == 'esm':
				self.longest_length = 238
			else:
				self.longest_length = 714
		elif drug == 'covid':
			self.longest_length = 1000	
		else:
			raise Exception("Longest length not defined")
	
		if generate_pregenerated:
			self.pre_generate_sequences()

	def get_number_loci(self):
		if self.drug == 'gfp' or self.drug == 'covid':
			return 1
		else:
			return len(self.regions)

	def return_train_strains(self):
		return self.train_strains

	def initialize_encoder(self):
		if not self.sequence_generation or self.frozen:
			all_train_outputs = [self.__getitem__(i) for i in tqdm(self.return_train_strains(), total=len(self.train_strains))]
			_, pre_defined_encoding_vector = self.return_encoding_barcode(all_train_outputs)
			self.pre_defined_encoding_vector = pre_defined_encoding_vector
			self.pre_defined_encoding_vector.sort()
			if self.frozen:
				self.id_to_tokens = None
				self.barcode_to_embedding = self.initialize_embedding()

		elif self.sequence_generation and not self.longest_length:
			self.longest_length = self.get_longest_length() 
			print(f"Found the longest length of a sequence is {self.longest_length}")
		elif self.esm_embedding and not self.frozen:
			if self.drug == 'INH' or self.drug == 'RIF' or self.drug == 'PZA':
				import pickle
				with open(f'/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/{self.drug}_tokens.pickle', 'rb') as handle:
					self.id_to_tokens = pickle.load(handle)
			else:
				self.id_to_tokens = None
				model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
				self.batch_converter = self.alphabet.get_batch_converter()
			
	

	def get_longest_length(self):
		longest_length = 0

		for strain in tqdm(self.return_train_strains(), total = len(self.train_strains)):
			output = self.__getitem__(strain)
			for sequence in output[0]:
				if sequence:
					if len(sequence) > longest_length:
						longest_length = len(sequence)

		return longest_length
			
	def get_len_vector(self):
		return len(self.pre_defined_encoding_vector)
	
	def return_test_strains(self):
		return self.test_strains

	def initialize_embedding(self):
		if self.embedding_file:
			embedding_path = self.embedding_file      
		elif self.drug == 'gfp':
			if self.unirep_embedding:
				embedding_path = "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/GFP/data/gfp_unirep_embeddings.pickle"
			elif self.esm_embedding:
				embedding_path = "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/GFP/data/gfp_esm2_embeddings_finetune.pickle"
		elif self.drug == 'covid':
			embedding_path = "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/Covid/data/covid_esm2b_embeddings.pickle"
		else:
			raise Exception("Embedding file is not coded in")

		import pickle

		with open(embedding_path, 'rb') as handle:
			barcode_to_embedding = pickle.load(handle)

		return barcode_to_embedding

	def return_barcode_embedding(self, get_item_outputs):
		encoded_output = []
		for output in get_item_outputs:
			to_add = []
			barcode = output[0]
			to_add.append(np.array(self.barcode_to_embedding[barcode]))
			to_add = to_add + output
			encoded_output.append(to_add)
			
		return encoded_output[0]

	def return_encoding_barcode(self, get_item_outputs, pre_defined_encoding_vector = False):
		"""

		Get item output is: [sequences, self.strain_to_label[i], i]
		Encodes input for barcode based models

		"""
		if not pre_defined_encoding_vector:
			all_mutations = set()
			for output in tqdm(get_item_outputs, total=len(get_item_outputs)):
				if output[0].rstrip():
					for mutation in output[0].split('-'):
						all_mutations.add(mutation)

			pre_defined_encoding_vector = list(all_mutations)

		encoded_output = []

		for output in get_item_outputs:
			to_add = []
			barcode = output[0]
			to_add.append([1 if mut in barcode else 0 for mut in pre_defined_encoding_vector])
			to_add = to_add + output
			encoded_output.append(to_add)

		return encoded_output[0], pre_defined_encoding_vector


	def return_encoding_nucleotide(self, get_item_outputs):
		"""
		Encodes input for nucleotide based models

		For CNN have to find longest loci, encode everything with respect longest loci
		Dimension of Input has to be Batch x Number Nucleotides x Longest Loci x Number Loci
		

		"""


		encoded_output = []

		def one_hot_encode(seq):
			if self.nucleotide:	
				mapping = dict(zip("ACGTX", range(5)))
			else:
				mapping = dict(zip("AFDGEVPNQTRCYLMWHSIK", range(20)))

			#At this point if sequence is amino acid, we got to convert to Nucleotide
			#For the CNN, to do this we just choose codons for those that are ambiguous
			#Should not affect model performance since performance conserved

			if self.drug == 'covid' or self.drug == 'gfp':
				codons = [amino_acid_to_codon[i] for i in seq]
				seq = ''.join(codons)

			if seq:
				seq2 = [mapping[i] for i in seq]
			else:
				seq2 = []

			#Here we pad input where we pad with the number 4 (0,1,2,3 is NN, 4 is pad)
			if len(seq2) < self.longest_length:
				seq2 = seq2 + [4]*(self.longest_length - len(seq2))
				
			return np.eye(5)[seq2]

		for output in get_item_outputs:
			to_add = [np.array([one_hot_encode(i) for i in output[0]])] + output
			encoded_output.append(to_add)

		return encoded_output


	def breakdown_regions(self, region, drug):
		result = {}
		for line in open(region, 'r').readlines()[1:]:
			data = line.split('\t')
			if data[0] == drug:
				result[data[1]] = [int(data[2]), int(data[3]), str(data[4])]

		return result


	def breakdown_strains(self, filepath):
		strains = []
		print(filepath)
		covid_min = -9.101593596174292
		covid_max = 0

		for line in open(filepath,'r').readlines():
			strain, label = line.split('\t')
			if self.drug == "covid":
				label = float(label.rstrip())
				label = (np.log(label) - covid_min)/(covid_max - covid_min)	
				self.strain_to_label[strain] = label
			else:
				self.strain_to_label[strain] = float(label.rstrip())
			strains.append(strain)

		return strains


	def run_cryptic_strain(self, strain):
		first_command = f"minimap2 -ax asm10 --cs /n/data1/hms/dbmi/farhat/mm774/References/h37rv.fasta /n/data1/hms/dbmi/farhat/rollingDB/cryptic_output/{strain}/spades/contigs.fasta | awk '$1 ~ /^@/ || ($5 == 60)' > output_{strain}.sam"
		os.system(first_command)

		second_command = f"samtools view -bS output_{strain}.sam | samtools sort - > output_{strain}.bam"
		os.system(second_command)

		third_command = f"samtools index output_{strain}.bam"
		os.system(third_command)
		
		return f"/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/output_{strain}.bam"

	def remove_cryptic_strain(self, strain):
		first_command = f"rm output_{strain}.sam"
		os.system(first_command)

		second_command = f"rm output_{strain}.bam"
		os.system(second_command)

		third_command = f"rm output_{strain}.bam.bai"
		os.system(third_command)

	def get_sequences(self, strain, region_breakdown = False, bam_file_provided = False):
		sequences = []

		original_bam_file_provided = bam_file_provided

		if bam_file_provided and 'cryptic' in original_bam_file_provided:
			bam_file_provided = self.run_cryptic_strain(strain)

		if not self.regions:	
			return [self.get_sequence(strain, self.drug, pad = False)]

		for i in self.regions:
				start, end, strand = self.regions[i]
				if strand == 'R':
						strand = '-'
				else:
						strand = '+'

				gene = (start, end, strand, None)
				result = self.get_sequence(strain, i, pad = False)

				char_pass = False
				if result is not None:
						final_result = ''
						for j in result:
							if j in ['A','T','C','G','X']:
								final_result = final_result + j
							else:
								final_result = final_result + 'X'
						char_pass = True
						result = final_result

				if result is not None and char_pass:
						if region_breakdown:
							sequences.append([i, str(result)])
						else:
							sequences.append(str(result))
				else:
						if region_breakdown:
							sequences.append([i, None])
						else:
							sequences.append(None)

		if bam_file_provided and 'cryptic' in original_bam_file_provided:
			self.remove_cryptic_strain(strain)
		
		return sequences 

	def pre_generate_sequences(self, strains_with_bam = False):
		data_file = open(f'/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/sequences_{self.drug}', 'w')
		if not strains_with_bam:
			strains = get_strains_barcode_file(f"/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/mutational_barcodes_reg_{self.drug}")
			strains_with_bam = [[i, False] for i in strains]

		for strain, bam_file in strains_with_bam:
			print(f"{strain}\t{bam_file}")
			sequences = self.get_sequences_aligner(strain, region_breakdown = True, bam_file_provided = bam_file)
			if sequences:
				for region, sequence in sequences:
					data_file.write(f'{strain}\t{region}\t{sequence}\n')
			else:
				print(f"{strain} failed")
		data_file.close()

	def get_sequences_barcode(self, strain):
		return self.fetcher.barcode(strain)

	def __getitem__(self, i):
		if self.sequence_generation:
			sequences = self.get_sequences(i)
		else:
			sequences = self.get_sequences_barcode(i)

		#Below is idea for normal alignment util to resample    
		##If isolate has less than 10 regions we say isolate is useless and throw it out
		if self.filter_data:
				good_quality_alignment = False

				while not good_quality_alignment:
						real_sequences = [i for i in sequences if i]
						if len(real_sequences) > 10:
								good_quality_alignment = True
						else:
								i = self.sampler.get_alternate_strain(i, 'train')
								print(f"NOT GOOD going again with {i}")
								if 'None' not in i:
										sequences = self.get_sequences(i)
		if self.random_mode:
			output = [sequences, random.choice([0,1]), i]
		else:
			output =  [sequences, self.strain_to_label[i], i]

		if self.pre_defined_encoding_vector and not self.sequence_generation:

			sample_barcode = self.return_encoding_barcode([output], self.pre_defined_encoding_vector)[0]
			if not self.esm_embedding and not self.unirep_embedding:
				return sample_barcode
			else:
				return self.return_barcode_embedding([output])
			
		elif self.sequence_generation and self.longest_length:
			if not self.frozen and not self.esm_embedding:
				return self.return_encoding_nucleotide([output])[0]
			elif not self.frozen and self.esm_embedding:
				if self.id_to_tokens:
					tokenized_sequences = []
					for gene in self.id_to_tokens[i]:
						tokenized_sequences.append(self.id_to_tokens[i][gene])
				else:

					sequence_output = []
					sequence_output[:0] = output[0][0].replace('*', '')

					if len(sequence_output) < self.longest_length:
						sequence_output = sequence_output + ['<pad>']*(self.longest_length - len(sequence_output))

					data = self.tokenizer([('protein1',''.join(sequence_output))])

				if self.model_type == "esm_mod":
					if not self.id_to_tokens:
						return [data[0].numpy(), None, self.strain_to_label[i], i]
					else:
						if len(tokenized_sequences) < 8:
							padding = [0] + [1]*(len(tokenized_sequences[0][0])-2) + [2]
							to_add = np.array([padding]*(8-len(tokenized_sequences)))
							num = 8-len(tokenized_sequences)
							tokenized_sequences = np.append(np.array(tokenized_sequences), to_add.reshape((num, 1, 1026)), axis=0)
						return [tokenized_sequences, None, self.strain_to_label[i], i]	
				else:
					mask_points, mlm_labels = self.mask(data[0].numpy()) 
					return [data[0].numpy(), mask_points, mlm_labels,i, self.strain_to_label[i]]
		else:
			return output

	def collate_fn(self, batch):
		X = []
		Y = []
		samples = []

		if self.model_type == "esm":
			task_label = []
				

		for i in batch:
			if self.id_to_tokens:
				X.extend(i[0])
			else:
				X.append(i[0])
			Y.append(i[2])
			samples.append(i[3])
			if self.model_type == "esm":
				task_label.append(i[4])

		final_batch = {}
		if not self.frozen and self.esm_embedding and self.model_type != 'esm_mod':
			final_batch['features'] = torch.LongTensor(X)
			final_batch['labels'] = [torch.LongTensor(Y), torch.FloatTensor(task_label)]
		else:
			if self.model_type == "esm_mod":
				if self.id_to_tokens:
					final_batch['features'] = torch.LongTensor(X).squeeze()
				else:
					final_batch['features'] = torch.LongTensor(X)

			else:
				final_batch['features'] = torch.FloatTensor(X)
			final_batch['labels'] = torch.FloatTensor(Y)
		final_batch['strains'] = samples
		return final_batch



	def train_dataloader(self, sampler):
		batch_sampler = torch.utils.data.sampler.BatchSampler(
		sampler,
		batch_size = self.batch_size,
		drop_last = False
		)

		return DataLoader(self, num_workers = 0, collate_fn = self.collate_fn, batch_sampler = batch_sampler)

	def val_dataloader(self, sampler):

		batch_sampler = torch.utils.data.sampler.BatchSampler(
                	sampler,
                	batch_size = self.batch_size,
                	drop_last = False
                	)
			
		return DataLoader(self, num_workers = 0, collate_fn = self.collate_fn, batch_sampler = batch_sampler)
	
	#####SEQUENCE COLLECTING STUFF, ALIGNMENT UTIL IS NOW PREPROCESSING, NOT DYNAMIC######
	def breakdown_sequence(self):
		if self.drug == 'gfp':
			sequence_file = f"/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/GFP/data/gfp_sequences"
		elif self.drug == 'covid':
			sequence_file = f"/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/Covid/data/covid_sequences"
		else:
			sequence_file  = f"/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/sequences_{self.drug}"
		
		strain_to_sequence = {}

		for line in open(sequence_file, 'r').readlines():
			data = line.split('\t')
			strain = data[0]
			if self.drug == 'gfp' or self.drug == 'covid':
				sequence = data[1].rstrip()
				gene = self.drug
			else:
				gene = data[1].rstrip()
				sequence = data[2].rstrip()
	
			if strain not in strain_to_sequence:
				strain_to_sequence[strain] = {}
			if gene not in strain_to_sequence[strain]:
				strain_to_sequence[strain][gene] = ''
			strain_to_sequence[strain][gene] = sequence

		return strain_to_sequence


	def get_sequence(self, sample, gene, pad = False):
		pot_sequence = self.strain_to_sequence[sample][gene]

		if len(pot_sequence) < self.longest_length and pad:
			pot_sequence = pot_sequence + 'X'*(self.longest_length - len(pot_sequence))
		return pot_sequence

	#####THIS IS NOW LANGUAGE MODEL PRETRAINING STUFF######

	def mask(self, np_array):
		#Mask rate is set to 15%
		self.mask_rate = 0.15
		mask_probs = np.random.random(np_array.shape)
		
		#This is random mask creations, where prob is less than mask_rate randomly add masks
		mask_flags  = (mask_probs < self.mask_rate)
		mask_encoded = '<mask>'
		mask_fills  = [mask_encoded]*np_array.shape[0]
		#Create mask points that are [MASK] A C K [MASK] where we want to mask
		mask_points = np.where(mask_flags, mask_fills, np_array)
		#Create a true label to compare that are A -100 -100 -100 A, to test if it predicts correctly
		mlm_labels  = np.where(mask_flags, np_array,  [-100]*np_array.shape[0])

		return mask_points, mlm_labels

	def tokenizer(self, data):
		batch_labels, batch_strs, batch_tokens = self.batch_converter(data) 
		return batch_tokens


codon_table = { 
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                  
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W', 
    }

amino_acid_to_codon = {}
for i in codon_table:
	codon = i
	amino_acid = codon_table[i]

	if amino_acid == '_':
		amino_acid = '*'

	if amino_acid not in amino_acid_to_codon:
		amino_acid_to_codon[amino_acid] = codon


