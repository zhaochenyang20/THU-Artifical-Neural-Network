# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = 
		self.bias = 

		# Store the average mean and variance
		self.register_buffer('running_mean', )
		self.register_buffer('running_var', )
		
		# Initialize your parameter

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		return input
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = 
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
