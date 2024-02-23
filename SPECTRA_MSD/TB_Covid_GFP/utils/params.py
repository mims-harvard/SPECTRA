from utils.constants import *

train_filepath =  "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/PZA_MUTATION_SPLIT_0_TRAIN"
test_filepath = "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/PZA_MUTATION_SPLIT_0_TEST"

dummy_filepath = "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/PZA_MUTATION_SPLIT_DUMMY"

save_location = "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/run/saved_runs/"

params_log_regression = {
        'run_name': 'baseline_log_reg_0',
        'alignment_location': ALIGNMENT_LOCATION ,
        'input_gff_file': GFF_FILE,
        'region': REGION_DATA,
        'batch_size': 256,
        'eval_size': EVAL_SIZE,
        'eval_freq': 5,
        'train_strains': train_filepath,
        'test_strains': test_filepath,
        'reference_nucleotide': REFERENCE_NUCLEOTIDE,
        'full_reference_sequence': FULL_REFERENCE_SEQUENCE,
        'drug':'PZA',
        'sequence_generation':False,
        'model_type': "logistic_regression",
        'lr': 0.1,
        'loss': "bce",
        'device':"cpu",
        'record_run':True,
        'save':True,
        'save_location': save_location,
        'save_freq': 5,
	'return_dataset': False
}

params_log_regression_covid = dict(params_log_regression)
params_log_regression_covid['lr'] = 0.001
params_log_regression_covid['l1_regularization'] = False
params_log_regression_covid['l1_regularization_param'] = 0.00

params_unirep = dict(params_log_regression)
params_unirep['unirep_embedding_generation'] = True
params_unirep['lr'] = 10**-4
params_unirep['l1_regularization'] = True
params_unirep['l1_regularization_param'] = 0.01 

params_esm = dict(params_log_regression)
params_esm['esm_embedding'] = True
params_esm['frozen'] = True
params_esm['lr'] = 10**-3
#params_esm['lr'] = 0.1
params_esm['l1_regularization'] = False
params_esm['l1_regularization_param'] = 0.00


params_esm_embedding = dict(params_log_regression)
params_esm_embedding['esm_embedding'] = True
params_esm_embedding['frozen'] = False
params_esm_embedding['lr'] = 10**-3
#params_esm['lr'] = 0.1
params_esm_embedding['l1_regularization'] = False
params_esm_embedding['l1_regularization_param'] = 0.00
params_esm_embedding['sequence_generation'] = True
params_esm_embedding['batch_size'] = 16
params_esm_embedding['loss'] = 'cross_entropy'

params_esm_mod = dict(params_log_regression)
params_esm_mod['esm_embedding'] = True
params_esm_mod['frozen'] = False
params_esm_mod['lr'] = 10**-6
#params_esm['lr'] = 0.1
params_esm_mod['l1_regularization'] = False
params_esm_mod['l1_regularization_param'] = 0.00
params_esm_mod['sequence_generation'] = True
params_esm_mod['batch_size'] = 8
#params_esm_mod['batch_size'] = 10


params_WDNN = dict(params_log_regression)
params_WDNN['model_type'] = 'wdnn'
params_WDNN['lr'] = 0.001


params_CNN = dict(params_log_regression)
params_CNN['model_type'] = 'CNN'
params_CNN['lr'] = 10**-6
params_CNN['sequence_generation'] = True
params_CNN['device'] = 'gpu'
params_CNN['batch_size'] = 256

params_SimpleCNN = dict(params_log_regression)
params_SimpleCNN['model_type'] = 'simpleCNN'
params_SimpleCNN['lr'] = 10**-4
params_SimpleCNN['sequence_generation'] = True
params_SimpleCNN['device'] = 'gpu'
params_SimpleCNN['batch_size'] = 256


