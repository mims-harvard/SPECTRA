from model.model import *
import pytorch_lightning as pl
from dataset.Sequence_Dataset import *
from utils.callbacks import *
from utils.samplers import *
from utils.constants import *
import wandb
from pytorch_lightning import Trainer
from run.TrainingModule import *
import os
from utils.params import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import esm
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
import torch.nn as nn

def run_baseline(
	run_name:			str = "run_name",
        alignment_location:             str = "/example/filepath",
        input_gff_file:                 str = "/example/filepath",
        region:                         str = "/example/filepath",
        train_strains:                  str = "/example/filepath",
        test_strains:                   str = "/example/filepath",
        reference_nucleotide:           str = "/example/filepath",
	unirep_embedding:		bool = False,
	esm_embedding:    		bool = False,
	frozen:				bool = False,
        full_reference_sequence:        str = "/example/filepath",
        drug:                           str = "PZA",
        sequence_generation:            bool = False,
        model_type:                     str = "logistic_regression",
	batch_size:			int =  64,
	eval_size:			int = 64,
	eval_freq:			int = 10,
	lr:				float = 3e-4,
	loss:				str = "bce",
	device:				str = "cpu",
	record_run:			bool = False,
	save:				bool = False,
	save_location:			str = "/example/filepath",
	save_freq:			float = 5,
	return_dataset:			bool = False,
	random_mode:			bool = False,
	balanced_batches:		bool = False,
	lambda_param:			float = 0.1,
	use_pregenerated:		bool = False,
	balance_datasets:		bool = True,
	label_type:			str = "binary",
	l1_regularization:		bool = False,
	l1_regularization_param:	float = 0.0,
	number:				int = 0,
	num_genes:			int = 1,
        embedding_file:                 str = None,
        ):

       
        gpus = 1 if torch.cuda.is_available() else None

        #Initialize W&B to record runs
        if record_run and wandb.run is None:
                if 'covid' in run_name:
                        if 'esm' in model_type:
                                wandb.init(project='covid_esm', entity='yasha')
                        elif 'seqdesign' in model_type:
                                wandb.init(project='covid_seq', entity='yasha')
                        else:
                                wandb.init(project='covid_mutdatasplit', entity='yasha')
                elif 'gfp' in run_name:
                        if model_type == "CNN":
                                wandb.init(project='gfp_cnn', entity='yasha')
                        elif 'seqdesign' in model_type:
                                wandb.init(project='gfp_seq', entity='yasha')
                        elif 'esm' in model_type:
                                wandb.init(project='gfp_esm', entity='yasha')
                        else:
                                wandb.init(project='gfp_mutdatasplit', entity='yasha')
                elif 'RIF' in run_name and 'esm' in model_type:
                        wandb.init(project='RIF_esm', entity='yasha')
                elif 'PZA' in run_name and 'esm' in model_type:
                        wandb.init(project='PZA_esm', entity='yasha')
                elif 'INH' in run_name and 'esm' in model_type:
                        wandb.init(project='INH_esm', entity='yasha')
                else:
                        wandb.init(project='TB_mutdatasplit', entity='yasha')

                wandb.run.name = run_name
                to_report = wandb
                wandb.config.lambda_param = lambda_param
                wandb.config.drug = drug
                wandb.config.model = model_type
                wandb.config.unirep_embedding = unirep_embedding
                wandb.config.esm_embedding = esm_embedding
                wandb.config.frozen = frozen
                wandb.config.label_type = label_type
                wandb.config.l1_regularization = l1_regularization_param
                wandb.config.l1_regularization_used = l1_regularization
                wandb.config.lr = lr 	
        elif not record_run:
                to_report = None


        import random
        random_value = '_'+str(random.random()*1000000).split('.')[0]

        #Check if save path is already made
        if save and os.path.isdir(save_location + run_name + random_value):
                print(save_location + run_name)
                raise Exception("Save path is already made, please specify another one")
        elif save and not os.path.isdir(save_location + run_name + random_value):
                os.mkdir(save_location + run_name + random_value)

        #Initilize dataset
        data_params = {
                'alignment_location': alignment_location ,
                'input_gff_file': input_gff_file,
                'region': region,
                'batch_size': batch_size,
                'eval_size': 64,
                'train_strains': train_strains,
                'test_strains': test_strains,
                'reference_nucleotide': reference_nucleotide,
                'full_reference_sequence': full_reference_sequence,
                'drug':drug,
                'sequence_generation':sequence_generation,
		'random_mode':random_mode,
		'use_pregenerated': use_pregenerated,
		'unirep_embedding':unirep_embedding,
		'esm_embedding':esm_embedding,
		'frozen': frozen,
		'model_type': model_type,
                'embedding_file':embedding_file
        }

        sequence_dataset = Sequence_Dataset(**data_params)
        sequence_dataset.initialize_encoder()

        if return_dataset:
                return sequence_dataset

        #Model initialization
        if model_type == "logistic_regression":
                if esm_embedding and frozen:
                        model = LogisticRegression(input_dim = 1280, num_classes = 1)
                elif unirep_embedding and frozen:
                        model = LogisticRegression(input_dim = 5700, num_classes = 1)
                else:
                        model = LogisticRegression(input_dim = sequence_dataset.get_len_vector(), num_classes = 1)
        elif model_type == "FNN":
                if esm_embedding and frozen:
                        model = FNN(input_dim = 1280, num_classes = 1)
                elif unirep_embedding and frozen:
                        model = FNN(input_dim = 5700, num_classes = 1)
                else:
                        model = FNN(input_dim = sequence_dataset.get_len_vector(), num_classes = 1)
        elif model_type == "FNN_Larger":
                if esm_embedding and frozen:
                        model = FNN_Larger(input_dim = 1280, num_classes = 1)
                elif unirep_embedding and frozen:
                        model = FNN_Larger(input_dim = 5700, num_classes = 1)
                else:
                        model = FNN_Larger(input_dim = sequence_dataset.get_len_vector(), num_classes = 1)
        elif model_type == "esm":
                config = {
                          'vocab_size':20,
                          'hidden_size':1280,
                          'num_hidden_layers':3,
                          'num_attention_heads': 3,
                          'intermediate_size': 1280,
                          'max_position_embeddings': 513,
                          'output_hidden_states': True,
                          'return_dict':True,
                          'hidden_dropout_prob': 0
                }
                bert_config = BertConfig(**config) 
                esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

                for i in esm_model.layers[:30]:
                    for param in i.parameters():
                        param.requires_grad = False
      
                batch_converter = alphabet.get_batch_converter()
                mlm_head = None
                model = [esm_model, mlm_head, LogisticRegression(input_dim = 1280, num_classes = 1)]
        elif model_type == "esm_mod":
                esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                esm_model.lm_head = LogisticRegression(320, 1, esm_mod = True, num_gene = num_genes)
                batch_converter = alphabet.get_batch_converter()
                model = esm_model  
                model = nn.DataParallel(model)           
        elif model_type == "BERT":
                config = {
                          'vocab_size':20,
                          'hidden_size':encoder_hidden_size,
                          'num_hidden_layers':encoder_num_hidden_layers,
                          'num_attention_heads': encoder_num_attention_heads,
                          'intermediate_size': encoder_intermediate_size,
                          'max_position_embeddings': 513,
                          'output_hidden_states': True,
                          'return_dict':True,
                          'hidden_dropout_prob': 0
                }

                bert_config = BertConfig(**config)
                BERT = BertModel(config)
                mlm_head = BertOnlyMLMHead(bert_config)
                model = [BERT, mlm_head, LogisticRegression(input_dim = 513, num_classes = 1)]
        elif model_type == "wdnn":
                model = WDNN(sequence_dataset.get_len_vector())
        elif model_type == "CNN":
                model = CNN(sequence_dataset.get_number_loci(), drug)
        elif model_type == "simpleCNN":
                model = SimpleCNN(sequence_dataset.get_number_loci(), drug)
        elif model_type == "unirep_embedding":
                model = LogisticRegression(input_dim = 5700, num_classes = 1)
        elif model_type == "esm_embedding":
                model = FNN(input_dim = 1280, num_classes = 1)
        elif model_type == "seqdesign_embedding":
                model = LogisticRegression(input_dim = 48, num_classes = 1)

        #Initialize sampler
        sampler_params = {
                'train_file_path': train_strains,
                'test_file_path': test_strains,
		'drug': drug,
		'balance': balance_datasets	
        }

        if balanced_batches:
                sampler = BalancedBarcode(drug, train_strains)
        else:
                if label_type == "categorical":
                        if drug == 'covidd':
                            sampler = CovidSampler(train_strains)
                        else:	
                            sampler_params['bin_sampler'] = False		
                            sampler = SimpleSamplerCategorical(**sampler_params)
                else:
                        sampler = SimpleSampler(**sampler_params)
	
        test_sampler = TestSampler(**sampler_params) 
        hparams = {'lr': lr, 'loss':loss, 'record':to_report,
		'model_type': model_type,
		'l1_regularization_param': l1_regularization_param,
		'l1_regularization': l1_regularization }

        model_module = TrainingModule(model, **hparams)

        #Add in relevant callbacks
        callbacks = []

        if save:
                start = time.time()
                
                callbacks.append(SaveCallback(save_freq, save_location + run_name + random_value ,start, sequence_dataset, model_type))
                if label_type == 'categorical':		
                        callbacks.append(ModelCheckpoint(dirpath=save_location + run_name + random_value, save_top_k = 2, monitor="val_spearman_rank", mode = 'max'))
                else:
                        callbacks.append(ModelCheckpoint(dirpath=save_location + run_name + random_value, save_top_k = 2, monitor="val_auc", mode = 'max')) 

        if label_type == 'categorical':
                if model_type == "esm" or model_type == "esm_mod":
                    if lambda_param >= 0 and lambda_param <= 0.2:
                        callbacks.append(EarlyStopping(monitor="val_spearman_rank", min_delta=0.00, patience=20, verbose=False, mode="max"))
                    else:
                        callbacks.append(EarlyStopping(monitor="val_spearman_rank", min_delta=0.00, patience=60, verbose=False, mode="max"))
                elif (model_type == "CNN" and drug == "gfp") or (model_type == "esm_embedding" and drug == "covid"):
                    callbacks.append(EarlyStopping(monitor="val_spearman_rank", min_delta=0.00, patience=60, verbose=False, mode="max"))
                else:
                    callbacks.append(EarlyStopping(monitor="val_spearman_rank", min_delta=0.00, patience=20, verbose=False, mode="max"))                
        else:
                callbacks.append(EarlyStopping(monitor="val_auc", min_delta=0.00, patience=20, verbose=False, mode="max"))

        #Initialize trainer and begin training
        trainer = Trainer(gpus=gpus,callbacks = callbacks, num_sanity_val_steps=0)

        trainer.fit(model_module, 
	train_dataloaders = sequence_dataset.train_dataloader(sampler), 
       	val_dataloaders = sequence_dataset.val_dataloader(test_sampler))

        sampler_params['type_sample'] = 'test'


        trainer.test(ckpt_path = "best", dataloaders = sequence_dataset.val_dataloader(TestSampler(**sampler_params)))


if __name__ == "__main__":	
	os.environ["WANDB_START_METHOD"] = "thread"

	import argparse

	parser = argparse.ArgumentParser("Run Baseline Model on Mutational Split")
	parser.add_argument("number", help="Which Split To Run On", type=int)
	parser.add_argument("drug", help="Which Drug To Run On", type=str)
	parser.add_argument("model", help="Which Model To Run On", type=str)
	parser.add_argument("lambda_param", help="Lambda Parameter", type=str)
	parser.add_argument("label_type", help = "binary or categorical", type=str)
	parser.add_argument("--random", help="randomly shuffle labels")
	parser.add_argument("--balanced_batches",help="fix barcode class imabalance")
	parser.add_argument("--trial_run", help="True if you want run to be unrecorded")
	parser.add_argument("--modified", help="Run on a modified split")
	args = parser.parse_args()

	params_to_use = None

	model = args.model
	if model == "logistic_regression" and args.drug != 'covid':
		params_to_use = params_log_regression
	elif model == "logistic_regression" and args.drug == 'covid':
		params_to_use = params_log_regression_covid
	elif model == "CNN":
		params_to_use = params_CNN
	elif model == "wdnn":
		params_to_use = params_WDNN
	elif model == "simpleCNN":
		params_to_use = params_SimpleCNN
	elif model == "unirep_embedding":
		params_to_use = params_unirep
	elif model == "esm_embedding" or model == "seqdesign_embedding":
		params_to_use = params_esm
	elif model == 'esm' or model == 'BERT':
		params_to_use = params_esm_embedding
	elif model == "esm_mod":
		params_to_use = params_esm_mod
		if args.drug == "covid":
			params_to_use['batch_size'] = 32
	else:
		raise Exception("Model type not recognized")

	lambda_param = float(str(float(args.lambda_param)*0.05)[:4])
	label_type = args.label_type
	
	dfp = "/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/TB_Resistance_Prediction/data/"
	train_file_path = dfp + f"{args.drug}_{label_type}_mutational_split/{args.drug}_{lambda_param}_MUTATION_SPLIT_{args.number}_TRAIN"
	test_file_path = dfp + f"{args.drug}_{label_type}_mutational_split/{args.drug}_{lambda_param}_MUTATION_SPLIT_{args.number}_TEST"
	if model == "seqdesign_embedding":
		embedding_file = f"/n/data1/hms/dbmi/zitnik/lab/users/ye12/TB_R_P/Utilities/SeqEmbeddings/{args.drug}_0.0_{args.number}_embeddings.pickle"
		params_to_use['embedding_file'] = embedding_file
	params_to_use['model_type'] = model
	params_to_use['label_type'] = label_type

	if args.modified:
		train_file_path = train_file_path + '_modified'
		test_file_path = test_file_path + '_modified'
	elif model == "esm":
		train_file_path = dfp + f"{args.drug}_{label_type}_mutational_split/test_gfp_TRAIN"
		test_file_path = dfp + f"{args.drug}_{label_type}_mutational_split/test_gfp_TEST"


	params_to_use['train_strains'] = train_file_path
	params_to_use['test_strains'] = test_file_path
		
	from datetime import datetime
	now = datetime.now()
	dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
	run_name = f'baseline_{model}_{lambda_param}_{args.drug}_{args.number}_{dt_string}'
	if args.random:
		params_to_use['run_name'] = run_name + '_random'
	elif args.balanced_batches:
		params_to_use['run_name'] = run_name + '_balanced'
	else:
		params_to_use['run_name'] = run_name

	params_to_use['record_run'] = True
	params_to_use['save'] = True

	if args.trial_run:
		params_to_use['record_run'] = False
		params_to_use['save'] = False

	params_to_use['use_pregenerated'] = True

	params_to_use['drug'] = args.drug 
	if args.drug == 'INH':
		params_to_use['num_genes'] = 14 
	elif args.drug == 'RIF':
		params_to_use['num_genes'] = 8
	elif args.drug == 'PZA':
		params_to_use['num_genes'] = 12

 
	params_to_use['random_mode'] = args.random
	params_to_use['balanced_batches'] = args.balanced_batches 
	params_to_use['lambda_param'] = lambda_param
	
	if label_type == 'categorical' and model != 'esm':
		params_to_use['loss'] = 'mae'

	params_to_use['number'] = args.number

	run_baseline(**params_to_use)