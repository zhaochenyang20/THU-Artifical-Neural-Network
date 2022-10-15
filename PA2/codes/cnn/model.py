# -*- coding: utf-8 -*-
<<<<<<< HEAD

from distutils.sysconfig import get_makefile_filename
=======
>>>>>>> 6933f2856d3b6741d1e45f15943b68bd7bd2fd47
import torch
from torch import nn
from torch.nn.parameter import Parameter
import config

class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum_1=0.8, momentum_2=0.9, eposilon=1e-5):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features
		self.momentum_1 = momentum_1
		self.momentum_2 = momentum_2
		self.eposilon = eposilon
		self.weight = Parameter(torch.ones(num_features))
		self.bias = Parameter(torch.zeros(num_features))
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			miu = input.mean([0, 2, 3])
			sigma2 = input.var([0, 2, 3])
			self.running_mean = 0.9 * self.running_mean + 0.1 * miu
			self.running_var = 0.9 * self.running_var + 0.1 * sigma2
		else:
			miu = self.running_mean
			sigma2 = self.running_var
		output = (input - miu[:, None, None]) / torch.sqrt(sigma2[:, None, None] + 1e-5)
		output = self.weight[:, None, None] * output + self.bias[:, None, None]
		return output
	# TODO END

class Dropout(nn.Module):
    # TODO START
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input):
        # input: [batch_size, num_feature_map, height, width]
        dp = torch.bernoulli(torch.ones(
            size=(input.shape[0], input.shape[1], 1, 1))*(1-self.p)).to(input.device)
        if not self.training:
            return input
        else:
            return dp*input/(1-self.p)
    # TODO END


class Model(nn.Module):
    def __init__(self, drop_rate=0.5):
        super(Model, self).__init__()
        # TODO START
        # Define your layers here
        self.model_list_1 = nn.ModuleList([
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
        ])
        self.model_list_2 = nn.ModuleList(
            [nn.Linear(config.output_feature_channel, 10)])
        # TODO END
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # TODO START
        # the 10-class prediction output is named as "logits"
        for sub_module in self.model_list_1:
            x = sub_module(x)
        x = x.reshape(x.shape[0], -1)
        for sub_module in self.model_list_2:
            x = sub_module(x)
        logits = x
        # TODO END

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        loss = self.loss(logits, y)
        correct_pred = (pred.int() == y.int())
        # Calculate the accuracy in this mini-batch
        acc = torch.mean(correct_pred.float())

        return loss, acc