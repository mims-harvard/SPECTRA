import torch, torch.nn as nn, torch.optim as optim, numpy as np
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score, accuracy_score
from utils.metrics import *

class TrainingModule(LightningModule):
	def __init__(
		self,
		transformer =  None,
		lr = 1e-4,
		loss = 'bce',
		record = False,
		l1_regularization = False,
		l1_regularization_param = 0,
		model_type = 'logistic_regression',
	):
		super().__init__()
		self.transformer = transformer
		self.lr = lr
		self.loss_str = loss
		if loss == 'bce':
			self.loss_function = nn.BCELoss()
		elif loss == 'kl':
			self.loss_function = nn.KLDivLoss(reduction='batchmean')
		elif loss == 'mae':
			self.loss_function = nn.L1Loss()
		elif loss == 'cross_entropy':
			self.loss_function = nn.CrossEntropyLoss()
		
		self.previous_loss = 0
		self.record = record
		self.l1 = l1_regularization
		self.lambda1 = l1_regularization_param
		self.model_type = model_type
	
		if model_type == "esm" or model_type == "BERT":
			self.encoder = self.transformer[0]
			self.encoding_top = self.transformer[1]
			self.task_top = self.transformer[2]
			self.automatic_optimization = False

	def forward(
		self,
		batch
	):

		if torch.cuda.is_available():
			batch = batch.cuda()
		if self.model_type == "esm":
			output = self.encoder(batch, repr_layers=[33])
			task_output = self.task_top(output['representations'][33].mean(axis=1).detach())
			output = [output['logits'], task_output]
		elif self.model_type == "BERT":
			encoder_output = self.encoder(batch)
			output = self.encoding_top(encoder_output['logits'])
		else:
			output = self.transformer(batch)
		
		if self.model_type == "esm_mod":
			output =  output['logits']             
  
		return output
	
	def loss(
		self,
		output,
		labels
	):
		if self.model_type == "esm":
			task_output = output[1]
			output = output[0]
			
			task_labels = labels[1] 
			labels = labels[0]

		if self.loss_str == 'bce':
			print(output.view(-1))
			print(labels)
			loss = self.loss_function(output.view(-1), labels)
		elif self.loss_str == "cross_entropy":
			loss = self.loss_function(output.view(-1, 33), labels.view(-1))
			task_loss = nn.BCELoss()(task_output.view(-1), task_labels)
			print(task_output.view(-1))
			print(task_labels)
			loss = [loss, task_loss]
			return loss
		else:
			print(output)
			print(labels)
			loss = self.loss_function(labels, output.squeeze())
		l1_regularization = 0
		if self.model_type == 'logistic_regression' and self.l1:
			linear_params = torch.cat([x.view(-1) for x in self.transformer.linear.parameters()])
			l1_regularization = self.lambda1 * torch.norm(linear_params, 1)
			
		return loss + l1_regularization

	def step(self, batch):
		data = batch['features']
		labels = batch['labels']
		strains = batch['strains']
		
		
		forward_out = self.forward(data)
		
		loss = self.loss(forward_out, labels)
		
		if self.loss_str == 'mae':
			logs = return_categorical_metrics_normal(labels.view(-1).detach().cpu().numpy(), forward_out.view(-1).detach().cpu().numpy(),strains, 'train')
			logs['train_loss'] = loss

		elif self.loss_str == "cross_entropy":
			logs = return_mlm_accuracy(labels[0], forward_out[0], 'train')
			task_logs = return_categorical_metrics_normal(labels[1].view(-1).detach().cpu().numpy(), forward_out[1].view(-1).detach().cpu().numpy(),strains, 'train')
			logs['train_loss'] = loss[0]
			logs['train_loss_task'] = loss[1]

			for i in task_logs:
				logs[i] = task_logs[i]

		else:
			logs = return_relevant_metrics(labels.view(-1).detach().cpu().numpy(), forward_out.view(-1).detach().cpu().numpy(), 'train')
			logs['train_loss'] = loss
 		
		
		return logs

	def training_step(self, batch, _):
		logs = self.step(batch)
		for k, v in logs.items():
			self.log(
        			k, v, on_step=True, on_epoch=False, prog_bar=True, logger=True,
          			sync_dist=True
            		)
		print(logs)
		if self.record:	
			self.record.log(logs)		

		if self.model_type == "esm":
			optimizer_language_model, optimizer_task_model = self.optimizers()
			
			####MANUAL OPTIMIZER STUFF####

			##LANGUAGE MODEL RUN##

			optimizer_language_model.zero_grad()
			self.manual_backward(logs['train_loss'])
			optimizer_language_model.step()
	
		return logs['train_loss']

	def testing_step(self, batch, _):
		data = batch['features']
		labels = batch['labels']
		strains = batch['strains']

		if torch.cuda.is_available():
			forward_out = self.forward(data.cuda())
		else:
			forward_out = self.forward(data)		

		loss = self.loss(forward_out, labels)
		if self.loss_str == 'mae':
			logs = return_categorical_metrics_normal(labels.view(-1).detach().cpu().numpy(), forward_out.view(-1).detach().cpu().numpy(), strains)
			logs['test_loss'] = loss
		elif self.loss_str == "cross_entropy":
			logs = return_mlm_accuracy(labels[0], forward_out[0], 'test')
			task_logs = return_categorical_metrics_normal(labels[1].view(-1).detach().cpu().numpy(), forward_out[1].view(-1).detach().cpu().numpy(),strains, 'test')
			logs['test_loss'] = loss[0].detach().item()

			for i in task_logs:
				logs[i] = task_logs[i]
		else:
			logs = return_relevant_metrics(labels.view(-1).detach().cpu().numpy(), forward_out.view(-1).detach().cpu().numpy())
			logs['test_loss'] = loss
		print(logs)

		return forward_out, labels

	def epoch_end(self, outputs, stage_str):
		pass

	def training_epoch_end(self, outputs):
		return self.epoch_end(outputs, 'Train')

	def validation_step(self, batch, batch_idx):
		data = batch['features']
		labels = batch['labels']

		if torch.cuda.is_available():
			predicted_values = self.forward(data.cuda())
		else:
			predicted_values = self.forward(data)

		if self.model_type == 'esm':
			return [labels, predicted_values, batch['strains']]
		else:
			return [labels.view(-1).detach().cpu().numpy(), predicted_values.view(-1).detach().cpu().numpy(), batch['strains']]

	def return_log(self, outputs, type_output, record = False):
		final_labels = []
		final_predictions = []
		final_samples = []
		final_data = []

		if self.model_type == "esm":
			final_mlm = []
			final_label_mlm = []

		for chunk in outputs:
			
			if self.model_type == "esm":
				final_predictions.extend(chunk[1][1].view(-1).detach().cpu().numpy())
				final_mlm.extend(chunk[1][0])

				final_labels.extend(chunk[0][1].view(-1).detach().cpu().numpy())
				final_label_mlm.extend(chunk[0][0])

				final_samples.extend(chunk[2])

				for lab, pred, samp in zip(chunk[0][1], chunk[1][1], chunk[2]):
					final_data.append([samp, lab, pred])

			else:
				final_predictions.extend(chunk[1])
				final_labels.extend(chunk[0])

				final_samples.extend(chunk[2])

				for lab, pred, samp in zip(chunk[0], chunk[1], chunk[2]):
					final_data.append([samp, lab, pred])
				

		if record:
			columns = ["sample", "label", "prediction"]
			final_pred = self.record.Table(data=final_data, columns = columns)
			self.record.log({"final_val_predictions":final_pred})
			if self.loss_str != "mae" and self.loss_str != "cross_entropy":
				self.record.log({"roc":self.record.plot.roc_curve( final_labels, [[1-i, i] for i in final_predictions], labels = ['Susceptible', 'Resistant'])})
				self.record.log({"pr":self.record.plot.pr_curve( final_labels, [[1-i, i] for i in final_predictions], labels = ['Susceptible', 'Resistant'])})


		loss = self.loss(torch.Tensor(final_predictions), torch.Tensor(final_labels))
		if self.loss_str == "mae":
			final_log = return_categorical_metrics_normal(np.array(final_labels), np.array(final_predictions), np.array(final_samples), type_output)
		elif self.loss_str == "cross_entropy":
			final_log = mlm_accuracy(final_mlm, final_label_mlm, type_output)
			task_logs = return_categorical_metrics_normal(np.array(final_labels), np.array(final_predictions), np.array(final_samples), type_output)
			logs['test_loss'] = loss[0]
			logs['test_loss_task'] = loss[1]

			for i in task_logs:
				final_log[i] = task_logs[i]
		else:
			final_log = return_relevant_metrics(final_labels, final_predictions, type_output)

		if self.model_type != "esm":
			final_log[f'{type_output}_loss'] = loss
		else:
			final_log['train_loss'] = loss[0].detach().item()
			final_log['train_loss_task'] = loss[1].detach().item()

		return final_log

	def validation_epoch_end(self, outputs):
		final_log = self.return_log(outputs, 'val')
		if self.record:
			self.record.log(final_log)

		for k, v in final_log.items():
			self.log(k, v)

		print(final_log)

		return self.epoch_end(outputs, 'Valid')

	def test_step(self, batch, batch_idx):
		return self.validation_step(batch, batch_idx)


	def test_epoch_end(self, outputs):
		final_log = self.return_log(outputs, 'final_val', True)
		if self.record:
			for key in final_log:
				self.record.run.summary[key] = final_log[key]

		for k,v in final_log.items():
			self.log(k,v)

		print(final_log)	
	
		return final_log

	def configure_optimizers(self):
		if self.model_type == "esm":
			optimizer_language_model = optim.AdamW([i for i in self.encoder.parameters() if i.requires_grad], lr=self.lr)
			optimizer_task_model = optim.AdamW(self.task_top.parameters(), lr=self.lr)
			return optimizer_language_model, optimizer_task_model
		else:
			optimizer = optim.AdamW(self.parameters(), lr=self.lr)
			return optimizer
