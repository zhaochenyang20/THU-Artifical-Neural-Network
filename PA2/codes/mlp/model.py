import torch
from torch import nn
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
		#! 仅在训练时候更新 buffer，测试时直接用 buffer 即可
		if self.training:
			average, variance = torch.mean(input, dim=0), torch.var(input, dim=0)
			self.running_mean = self.momentum_1 * self.running_mean + (1 - self.momentum_1) * average
			self.running_var = self.momentum_2 * self.running_var + (1 - self.momentum_2) * variance
		else:
			average, variance = self.running_mean, self.running_var

		normalized_input = (input - average) / torch.sqrt(variance + self.eposilon) * self.weight + self.bias
		return normalized_input
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
		self.layers = nn.Sequential(
			[
				nn.Linear(config.num_features, config.hidden_neuron),
				BatchNorm1d(config.hidden_neuron),
				#! 一定用 ReLU 吗？
    			nn.ReLU(),
				Dropout(p = drop_rate),
				nn.Linear(config.hidden_neuron, config.num_classes),
			]
		)
		# TODO ENDgp;
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.layers(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc