# # -*- coding: utf-8 -*-

# from importlib.metadata import requires
# from socketserver import ThreadingMixIn
# import torch
# from torch import nn
# from torch.nn import init
# from torch.nn.parameter import Parameter
# import config
# class BatchNorm1d(nn.Module):
# 	# TODO START
# 	def __init__(self, num_features,eps=1e-5,mome1=0.8,mome2=0.9):
# 		super(BatchNorm1d, self).__init__()
# 		self.num_features = num_features
# 		self.mome1=mome1
# 		self.mome2=mome2
# 		self.eps=eps

# 		# Parameters
# 		self.weight = nn.Parameter(torch.ones(num_features))
# 		self.bias =nn.Parameter(torch.ones(num_features))

# 		# Store the average mean and variance
# 		self.register_buffer('running_mean', torch.ones(num_features))
# 		self.register_buffer('running_var', torch.ones(num_features))

# 		# Initialize your parameter

# 	def get_mean_and_var(self,inputs):
# 		return inputs.mean(dim=0),inputs.var(dim=0)

# 	def forward(self, input):
# 		# input: [batch_size, num_feature_map * height * width]
# 		training_mode=self.training

# 		if training_mode:
# 			mean_value,var_value=self.get_mean_and_var(input)
# 			self.running_mean=self.running_mean*self.mome1+(1-self.mome1)*mean_value
# 			self.running_var=self.running_var*self.mome2+(1-self.mome2)*var_value
# 			return (input-mean_value)/torch.sqrt(self.eps+var_value)*self.weight+self.bias

# 		else:
# 			return (input-self.running_mean)/torch.sqrt(self.eps+self.running_var)*self.weight+self.bias

# 	# TODO END

# class Dropout(nn.Module):
# 	# TODO START
# 	def __init__(self, p=0.5):
# 		super(Dropout, self).__init__()
# 		self.p = p

# 	def forward(self, input):
# 		# input: [batch_size, num_feature_map * height * width]
# 		training_mode=self.training
# 		if not training_mode:
# 			return input
# 		else:
# 			dp=torch.bernoulli(torch.ones_like(input)*(1-self.p))
# 			return dp*input/(1-self.p)
# 	# TODO END

# class Model(nn.Module):
# 	def __init__(self, drop_rate=0.5):
# 		super(Model, self).__init__()
# 		# TODO START
# 		# Define your layers here
# 		# TODO END

# 		self.model_list=nn.ModuleList([
# 			nn.Linear(config.input_size,config.num_feature1),
# 			BatchNorm1d(config.num_feature1),
# 			nn.ReLU(),
# 			Dropout(drop_rate),
# 			nn.Linear(config.num_feature1,config.num_feature2),
# 		])
# 		self.loss = nn.CrossEntropyLoss()

# 	def forward(self, x, y=None):
# 		# TODO START
# 		# the 10-class prediction output is named as "logits"
# 		for sub_module in self.model_list:
# 			x=sub_module(x)
# 		logits = x
# 		# TODO END

# 		pred = torch.argmax(logits, 1)  # Calculate the prediction result
# 		if y is None:
# 			return pred
# 		loss = self.loss(logits, y.long())
# 		correct_pred = (pred.int() == y.int())
# 		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

# 		return loss, acc

# -*- coding: utf-8 -*-

from locale import normalize
from statistics import variance
from numpy import average
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class Config():
	def __init__(self, batch_size=100, hidden_neuron=100, num_epochs=20, learning_rate=1e-5, drop_rate=0.5, \
     ):
		self.batch_size = batch_size
		self.hidden_neuron = hidden_neuron
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.drop_rate = drop_rate
		self.num_classes = 10
		self.num_features = 3072

class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum_1=0.8, momentum_2=0.9, eposilon=1e-5):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features
		self.momentum_1 = momentum_1
		self.momentum_2 = momentum_2
		self.eposilon = eposilon

		# Parameters
		#! different torch.ones 的参数
		self.weight = Parameter(torch.ones(num_features))
		self.bias = Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		#! register_buffer 的作用是什么，running_mean 和 running_var 的意义
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))

		# Initialize your parameter

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		#! TODO 为什么是 training 不是 train
  		#! 这里我用的是 torch 的 mean 和 variance
		if self.training:
			average, variance = torch.mean(input, dim=0), torch.var(input, dim=0)
			self.running_mean = self.momentum_1 * self.running_mean + (1 - self.momentum_1) * average
			self.running_var = self.momentum_2 * self.running_var + (1 - self.momentum_2) * variance
			normalize_output = (input - average) / torch.sqrt(variance + self.eposilon)*self.weight + self.bias
			return normalize_output
		else:
			normalize_output = (input - self.running_mean) / torch.sqrt(self.running_var + self.eposilon)*self.weight + self.bias
			return normalize_output

	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			#! 这里和 Lafite 实现也不太一样
			dropout_distribution = torch.bernoulli(torch.ones_like(input) * (1 - self.p))
			return input * dropout_distribution / (1 - self.p)
		else:
			return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		#! nn.Sequential 和 nn.ModuleList 区别
		config = Config()
		self.layers = nn.ModuleList(
			[
				nn.Linear(config.num_features, config.hidden_neuron),
				BatchNorm1d(config.hidden_neuron),
				#! 一定用 ReLU 吗？
    			nn.ReLU(),
				Dropout(p = drop_rate),
				nn.Linear(config.hidden_neuron, config.num_classes),
			]
		)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		for each_layer in self.layers:
			x = each_layer(x)
		logits = x
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc