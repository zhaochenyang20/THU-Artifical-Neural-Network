from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        def _EuclideanLoss(input, target):
            return np.sum((input - target)**2) / 2
        loss = _EuclideanLoss(input, target)
        return loss
        # TODO END

    def backward(self, input, target):
		# TODO START
        def _diriviation_EuclideanLoss(input, target):
            return input - target
        diriviation_loss = _diriviation_EuclideanLoss(input, target)
        return diriviation_loss / input.shape[0]
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        def _softmax_cross_entropy_loss(input, target):
            return -np.sum(target * np.log(input)) / input.shape[0]
        loss = _softmax_cross_entropy_loss(input, target)
        return loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        denominator = np.expand_dims(np.exp(input).sum(-1),
                                     axis=-1) * np.ones_like(input)
        numerator = np.exp(input)
        softmax = np.divide(numerator, denominator)
        batch_size = input.shape[0]
        difference = (softmax - target) / batch_size
        return difference
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START
        hinge = np.multiply(
            target != 1,
            np.maximum(
                0, self.margin + input -
                np.multiply(input[target == 1].reshape(-1, 1), target != 1)))
        loss = np.mean(np.sum(hinge, axis=-1), axis=-1)
        return loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        batch_size = input.shape[0]
        condition = (self.margin - input[target == 1].reshape(-1, 1) + input >
                     0) & (target != 1)
        grad = np.multiply(np.ones_like(input), condition)
        grad -= (target == 1) * np.sum(grad, axis=1, keepdims=True)
        grad /= batch_size
        return grad
        # TODO END
