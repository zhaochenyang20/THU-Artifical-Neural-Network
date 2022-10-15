# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn.parameter import Parameter

class Config():
    def __init__(self, batch_size=100, hidden_neuron=100, num_epochs=20, learning_rate=1e-5, drop_rate=0.5,  kernel_size1=5, kernel_size2=3, channel1=128, channel2=64,\
                 output_feature_channel=2304, max_pool_size=2):
        self.batch_size = batch_size
        self.hidden_neuron = hidden_neuron
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.drop_rate = drop_rate
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.output_feature_channel = output_feature_channel
        self.max_pool_size = max_pool_size
        self.channel1 = channel1
        self.channel2 = channel2

class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum_1=0.8, momentum_2=0.9, eposilon=1e-5):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features
		self.momentum_1 = momentum_1
		self.momentum_2 = momentum_2
		self.eposilon = eposilon
        #! torch.nn.parameter https://zhuanlan.zhihu.com/p/344175147
		self.weight = Parameter(torch.ones(num_features))
		self.bias = Parameter(torch.zeros(num_features))
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			mean = input.mean([0, 2, 3])
			variance = input.var([0, 2, 3])
			self.running_mean = self.momentum_1 * self.running_mean + (1 -  self.momentum_1) * mean
			self.running_var = self.momentum_2 * self.running_var + (1 - self.momentum_2) * variance
		else:
			mean = self.running_mean
			variance = self.running_var

		normalized_input = (input - mean[:, None, None]) / torch.sqrt(variance[:, None, None] + self.eposilon)
		denormalized_input = self.weight[:, None, None] * normalized_input + self.bias[:, None, None]
		return denormalized_input
	# TODO END

class Dropout(nn.Module):
    # TODO START
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input):
        # input: [batch_size, num_feature_map, height, width]
        dp = torch.bernoulli(torch.ones(
            size = (input.shape[0], input.shape[1], 1, 1)) * (1 - self.p)).to(input.device)
        if not self.training:
            return input
        else:
            return dp * input / (1 - self.p)
    # TODO END


class Model(nn.Module):
    def __init__(self, drop_rate=0.5):
        super(Model, self).__init__()
        # TODO START
        config = Config()
        # Define your layers here
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=config.channel1,
                      kernel_size=config.kernel_size1),
            BatchNorm2d(config.channel1),
            nn.ReLU(),
            Dropout(drop_rate),
            nn.MaxPool2d(config.max_pool_size),
            nn.Conv2d(in_channels=config.channel1,
                      out_channels=config.channel2, kernel_size=config.kernel_size2),
            BatchNorm2d(config.channel2),
            nn.ReLU(),
            Dropout(drop_rate),
            nn.MaxPool2d(config.max_pool_size),
        )
        self.classify = nn.Linear(config.output_feature_channel, 10)
        # TODO END
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # TODO START
        # the 10-class prediction output is named as "logits"
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        logits = self.classify(x)
        # TODO END

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        loss = self.loss(logits, y)
        correct_pred = (pred.int() == y.int())
        # Calculate the accuracy in this mini-batch
        acc = torch.mean(correct_pred.float())

        return loss, acc