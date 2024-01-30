from utils.constants import *
from tqdm import tqdm

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

class Generate_Train_Test_GENERAL():

	def __init__(self, train_strains, test_strains, organism):
		self.train_strains = train_strains
		self.test_strains = test_strains
		self.organism = organism
		if 'spike' not in organism:
			self.resistance_mapping = self.generate_resistance_mapping()

	def generate_resistance_mapping(self):
		phenotype_mapping = {}

		phenotype_to_analyze = None
		if self.organism == 'covid':
			phenotype_to_analyze = COVID_CATEGORICAL_PHENOTYPES 
		elif self.organism == 'gfp':
			phenotype_to_analyze = GFP_CATEGORICAL_PHENOTYPES

		for line in open(phenotype_to_analyze, 'r').readlines():
			data = line.split('\t')
			phenotype_mapping[data[0]] = data[1].rstrip()

		return phenotype_mapping

	def generate_train_test(self, name_file):

		path_to_data = "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/"
		train_file = open(path_to_data+name_file+'_TRAIN', 'w')
		test_file = open(path_to_data+name_file+'_TEST', 'w')

		for strain in self.train_strains:
			if 'spike' not in self.organism:
				train_file.write(f'{strain}\t{self.resistance_mapping[strain]}\n')
			else:
				train_file.write(f'{strain}\n')

		train_file.close()

		for strain in self.test_strains:
			if 'spike' not in self.organism:
				test_file.write(f'{strain}\t{self.resistance_mapping[strain]}\n')
			else:
				test_file.write(f'{strain}\n')

		test_file.close()



class Generate_Train_Test():
	"""
		Helper function that has mutiple modes.
		(1) if return all strains is True it will return all strains with phenotypic and genotypic data for the drug

		(2) if return all strains is False, file_provided is true it will filter a provided train test split in a file

		(3) if return all strains is False, file_provided is false it will filter based on provided strains

	"""

	def __init__(self, train_strains, test_strains, data_location, file_provided = True, return_all_strains = False, skip_vcf = False):


		if return_all_strains is False:
			if file_provided:
				self.train_strains = [i.rstrip() for i in open(train_strains, 'r').readlines()]
				self.test_strains = [i.rstrip() for i in open(test_strains, 'r').readlines()]
			else:
				self.train_strains = train_strains
				self.test_strains = test_strains

			print(f"Initial data split had {len(self.train_strains)} train strains and {len(self.test_strains)} test strains")

			self.data_location = data_location
			self.df = pd.read_csv(METADATA, sep='\t')
			self.df.index = self.df['strain_res']
			self.isolates_in_resistance = list(self.df['strain_res'].values)
			self.isolates_with_data = None
		else:
			self.data_location = data_location
			self.df = pd.read_csv(METADATA, sep='\t')
			self.df.index = self.df['strain_res']
			self.isolates_in_resistance = list(self.df['strain_res'].values)
			self.isolates_with_resistance = self.resistance_lookup(self.isolates_in_resistance, return_all_strains)
			self.isolates_with_data = [i[0] for i in self.isolates_with_resistance]
			self.generate_name_mapping()
			self.resistance_lookup(self.isolates_with_data, return_all_strains)

	def return_isolates_with_data(self):
		return self.isolates_with_data


	def check_vcf_file(self, strain, files_to_search):
		vcf_file_location = "/n/data1/hms/dbmi/farhat/ye12/zipped_indexed_vcf_files/"
		for i in files_to_search:
			potential_strain_name = i.split('.')[0]
			if strain in potential_strain_name and len(set(strain)) == len(set(potential_strain_name)):
				return True
		return False


	def return_strains_present_database(self, strains_to_check, skip_vcf = False):
		strains_present = []

		files_present = [i for i in os.listdir(self.data_location)]
		files_to_search = [i for i in os.listdir("/n/data1/hms/dbmi/farhat/ye12/zipped_indexed_vcf_files/")]
		for i in tqdm(strains_to_check, total=len(strains_to_check)):
			for j in files_present:
				if i in j:
					#import pdb; pdb.set_trace()
					if skip_vcf:
						strains_present.append(i)
						break
					elif self.check_vcf_file(i, files_to_search):
						strains_present.append(i)
						break	
		print(f"Started with {len(strains_to_check)} strains and ended up with {len(strains_present)} after filtering for data presense")		
		return strains_present

	def generate_name_mapping(self):
		name_mapping = {}

		for line in open(NAME_MAPPING, 'r').readlines()[1:]:
			data = line.split(',')
			name_mapping[data[0]] = [data[1], data[2].rstrip()]

		self.name_mapping = name_mapping
	
	def resistance_lookup(self, strains_check, drug):
		strains_in_resistance = []

		number_resistant = 0
		number_susceptible = 0
		number_no_data = 0

		for j in strains_check:
			j = j.replace('-', '')
	
			i  = False

			if j in self.isolates_in_resistance:
				i = j
			else:
				for pot_name in self.name_mapping[j]:
					if pot_name in self.isolates_in_resistance:
						i = pot_name
						break
				
			if i:
				resistance_value = self.df.at[i, drug]
				if resistance_value == 0 or resistance_value == 1:
					strains_in_resistance.append([j,  resistance_value])


				if resistance_value == 0:
					number_susceptible += 1
				if resistance_value == 1:
					number_resistant += 1
				if resistance_value == -1:
					number_no_data += 1

		
		print(f"Provided with {len(strains_check)} strains. Resistant: {number_resistant}. Susceptible: {number_susceptible}. No Data: {number_no_data}.")		


		return strains_in_resistance


	def get_summary_stats(self, data, name):
		num_resistant = 0
		num_susceptible = 0

		for strain, label in data:		
			if label == 1:
				num_resistant += 1
			elif label == 0:
				num_susceptible += 1

		print(f"Stats for {name} is {num_resistant} resistant isolates and {num_susceptible} susceptible isolates")						
			
			
	def generate_train_test(self, name_file, drug, all_strains_present):
		if all_strains_present:
			strains_present = all_strains_present
		else:
			strains_present = self.return_strains_present_database(self.train_strains+self.test_strains)
		
		new_train = [i for i in self.train_strains if i in strains_present]
		new_test = [i for i in self.test_strains if i in strains_present]

		print(f"Post data filtering had {len(new_train)} train strains and {len(new_test)} test strains")
		new_train_r = self.resistance_lookup(new_train, drug)
		new_test_r = self.resistance_lookup(new_test, drug)

		print(f"Post resistance filtering had {len(new_train_r)} train strains and {len(new_test_r)} test strains")
		self.get_summary_stats(new_train_r, 'TRAIN')
		self.get_summary_stats(new_test_r, 'TEST')

		path_to_data = "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/"
		train_file = open(path_to_data+name_file+'_TRAIN', 'w')
		test_file = open(path_to_data+name_file+'_TEST', 'w')

		for strain, value in new_train_r:
			train_file.write(f"{strain}\t{value}\n")

		for strain, value in new_test_r:
			test_file.write(f"{strain}\t{value}\n")


	
