import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.nn.functional import softmax, sigmoid

class FeedForwardNetwork(nn.Module):
	def __init__(self, embedding_dim, activation_dropout):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.activation_fn = nn.GELU()
		self.activation_dropout_module = nn.Dropout(activation_dropout,)
		self.fc1 = nn.Linear(embedding_dim, 100)
		self.fc2 = nn.Linear(100, 1)

		self.fc3 = nn.Linear(embedding_dim, 1)

		self.m = nn.Sigmoid()

	
	def forward(self, x):
		x = self.activation_fn(self.fc1(x))
		x = self.activation_dropout_module(x)
		x = self.fc2(x)
		return self.m(x)

class LogisticRegression(nn.Module):
	def __init__(self, 
		input_dim:	int, 
		num_classes:	int,
                esm_mod:	bool = False,
		num_gene:	int = 1):

		super().__init__()
		self.esm_mode = esm_mod
		self.linear = nn.Linear(in_features=input_dim, out_features=num_classes, bias=True)
		self.num_gene = num_gene

	def forward(self, x):
		if self.esm_mode:
			x = x.mean(axis=1)
		print(x.shape)
		if self.num_gene > 1:
			i, j = x.shape
			x = x.reshape( int(i/self.num_gene) ,self.num_gene, j).mean(axis=1)
		x = self.linear(x) 
		y_hat = torch.sigmoid(x)
		return y_hat

class FNN(nn.Module):
	def __init__(self, input_dim, num_classes):
		super().__init__()
		self.linear_one = nn.Linear(in_features=input_dim, out_features=256, bias=True)
		self.linear_two = nn.Linear(in_features=256, out_features=num_classes, bias=True)
		#self.linear_three = nn.Linear(in_features = 512, out_features = 256, bias = True)
		#self.linear_four = nn.Linear(in_features=256, out_features=num_classes, bias=True)
	
	def forward(self, x):
		x = self.linear_one(x)
		x = self.linear_two(x)
		#x = self.linear_three(x)
		#x = self.linear_four(x)
		y_hat = torch.sigmoid(x)
		return y_hat

class FNN_Larger(nn.Module):
	def __init__(self, input_dim, num_classes):
		super().__init__()
		self.linear_one = nn.Linear(in_features=input_dim, out_features=1028, bias=True)
		self.linear_two = nn.Linear(in_features=1028, out_features=1028, bias=True)
		self.linear_three = nn.Linear(in_features = 1028, out_features = 512, bias = True)
		self.linear_four = nn.Linear(in_features=512, out_features=num_classes, bias=True)

	def forward(self, x):
		x = self.linear_one(x)
		x = self.linear_two(x)
		x = self.linear_three(x)
		x = self.linear_four(x)
		y_hat = torch.sigmoid(x)
		return y_hat


class MLMPretrainingHead(nn.Module):
	# Wraps a huggingface-style MLM PT head.
	def __init__(self, config, head_cnstr, mask_id):
		super().__init__()
		self.config = config
		self.head = head_cnstr(self.config)
		self.mask_id = mask_id

	def forward(
		self,
		points_encoded,
		point_kwargs
	):

		#import pdb;pdb.set_trace()
		sequence_out = points_encoded['granular']
		labels = point_kwargs['labels']

		prediction_scores = self.head(sequence_out)
		loss_fct = nn.CrossEntropyLoss()
		point_loss = loss_fct(
			prediction_scores.view(-1, self.config.vocab_size),
			labels.view(-1)
		)

		accuracy_overall, accuracy_masked = mlm_accuracy(
			point_kwargs,
			prediction_scores,
			self.mask_id
		)

		accuracy_overall = torch.Tensor([float(accuracy_overall)]).type_as(point_loss)
		accuracy_masked = torch.Tensor([float(accuracy_masked)]).type_as(point_loss)

		return {
			'loss': point_loss,
			'predictions': {
				'prediction_scores': prediction_scores,
				'predictions': prediction_scores.argmax(dim=2),
			},
			'metrics': {
				'accuracy_overall': accuracy_overall,
				'accuracy_masked': accuracy_masked,
			}
		}


"""
Wide and Deep Neural Network Routed Over From TensorFlow
As Described:

https://github.com/aggreen/MTB-CNN/blob/main/wdnn/wdnn_training.py


"""

class WDNN(nn.Module):

	def __init__(self, input_length):
		super().__init__()
		self.fc1 = nn.Linear(input_length,256)
		self.fc2 = nn.Linear(256,256)
		self.fc3 = nn.Linear(256,256)
		self.fc4 = nn.Linear(256,1)
		self.activation = F.ReLU()
		self.sigmoid = F.Sigmoid()
		self.batch_norm = nn.BatchNorm2d(256)
		self.dropout = nn.Dropout()



	def forward(self, x):
		#Initial Feature Massaging
		x = self.fc1(x)
		x = self.activation(x)
		x = self.batch_norm(x)
		x = self.dropout(x)
		#Block 1
		x = self.fc2(x)
		x = self.activation(x)
		x = self.batch_norm(x)
		x = self.dropout(x)
		#Block 2
		x = self.fc3(x)
		x = self.activation(x)
		x = self.batch_norm(x)
		x = self.dropout(x)
		#Output Block
		x = self.fc4(x)
		x = self.sigmoid(x)
		
		return x

"""
Convolutional Neural Network Routed Over From TensorFlow
As Described:

https://github.com/aggreen/MTB-CNN/blob/main/md_cnn/model_training/run_MDCNN_ccp_crossval.py

"""

class CNN(nn.Module):

	def __init__(self, number_of_loci, drug):
		super().__init__()
		
		self.conv2d_bn=nn.BatchNorm2d(64)
		self.conv1d_1_bn=nn.BatchNorm2d(64)
		self.conv1d_2_bn=nn.BatchNorm2d(32)
		self.conv1d_3_bn=nn.BatchNorm2d(32)
		self.fc1_bn=nn.BatchNorm1d(256)
		self.fc2_bn=nn.BatchNorm1d(256)

		#Conv2D with 64 filters, 5 by 5 kernel size
		self.conv2d = nn.Conv2d(number_of_loci, 64, (5,12))
		self.conv1d_1 = nn.Conv1d(64,64, (1,12))
		self.conv1d_2 = nn.Conv1d(64,32, (1,3))
		self.conv1d_3 = nn.Conv1d(32,32, (1,3))

		self.maxpool = nn.MaxPool1d(3)
		self.activation = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		#Hardcoded this
		if drug == 'PZA':
			self.fc1 = nn.Linear(10944,256, bias = True)
		elif drug == 'RIF':
			self.fc1 = nn.Linear(26720,256, bias = True)
		elif drug == 'INH':
			self.fc1 = nn.Linear(10240, 256, bias = True)
		elif drug == 'gfp':
			self.fc1 = nn.Linear(2400, 256, bias = True)
		elif drug == 'covid':
			self.fc1 = nn.Linear(3424, 256, bias = True)

		self.fc2 = nn.Linear(256,256, bias = True)
		self.fc3 = nn.Linear(256,1, bias = True)


	def forward(self, x):
		x = x.permute((0,1,3,2))
		#Initial Convolution Block
		x = self.activation(self.conv2d_bn(self.conv2d(x)))


		#Main Convolutional Block
		x = self.activation(self.conv1d_1_bn(self.conv1d_1(x)))
		x = self.maxpool(x.squeeze(2)).unsqueeze(2)
		x = self.activation(self.conv1d_2_bn(self.conv1d_2(x)))
		x = self.activation(self.conv1d_3_bn(self.conv1d_3(x)))
		x = self.maxpool(x.squeeze(2)).unsqueeze(2)
		x = torch.flatten(x, start_dim=1)


		#Output Block
		x = self.fc1(x)
		x = self.activation(self.fc1_bn(x))
		x = self.fc2(x)
		x = self.activation(self.fc2_bn(x))
		x = self.fc3(x)
		x = self.sigmoid(x)

		return x



class SimpleCNN(nn.Module):

	def __init__(self, number_of_loci, drug):
		super().__init__()
	
		self.conv2d = nn.Conv2d(number_of_loci, 64, (5,12))
		self.conv1d_1 = nn.Conv1d(64,64, (1,12))

		self.maxpool = nn.MaxPool1d(3)
		self.activation = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

		if drug == 'PZA':
			self.fc1 = nn.Linear(10944,256, bias = True)
		elif drug == 'RIF':
			self.fc1 = nn.Linear(160576,256, bias = True)
		elif drug == 'gfp':
			self.fc1 = nn.Linear(14720 ,256, bias = True)
		self.fc2 = nn.Linear(256,1, bias = True)

	def forward(self, x):
		x = x.permute((0,1,3,2))
		
		x = self.activation(self.conv2d(x))
		x = self.activation(self.conv1d_1(x))
		x = self.maxpool(x.squeeze(2)).unsqueeze(2)
		x = torch.flatten(x, start_dim=1)

		print(x.shape)
		x = self.fc1(x)
		x = self.activation(x)
		x = self.fc2(x)
		x = self.sigmoid(x)

		return x