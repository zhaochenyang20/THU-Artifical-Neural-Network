from __future__ import division
import re
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name
    # target 是 label 组成的 tensor
    def forward(self, input, target):
        # TODO START
        #*  loss input 是 batch_size * num_class 的矩阵，target 是 batch_size * num_class 的矩阵（one-hot）
        def _EuclideanLoss(input, target):
            return np.sum((input - target)**2) / 2
        loss = _EuclideanLoss(input, target) / input.shape[0]
        return loss
        # TODO END

    def backward(self, input, target):
		# TODO START
        def _diriviation_EuclideanLoss(input, target):
            return input - target
        diriviation_loss = _diriviation_EuclideanLoss(input, target)
        return diriviation_loss / input.shape[0]
		# TODO END


#! Loss 层不是就到头了吗，forward 给谁，从谁那儿 backard
#! Softmax is a way to recurrence, and cross-entropy is the way conjuated to compute loss
class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        #! reference: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
        #* 计算 softmax 前先减去最后一维的最大值
        input_shift = input - input.max(axis=-1, keepdims=True)
        exp_input = np.exp(input_shift)
        prob = exp_input / (exp_input.sum(axis=-1, keepdims=True))
        #* 为了避免梯度出现 NaN，需要做 clip
        prob_clip = prob.clip(min=1e-12, max=1 - 1e-12)
        #* 注意此处有了多个 batch，故而需要对 batch 内部的 loss 求均值
        #! TODO? 真的要求均值吗？
        corss_entropy = -(np.log(prob_clip) * target).sum(-1)
        batch_loss = np.mean(corss_entropy)
        return batch_loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        #! TODO
        batch_size = input.shape[0]
        input_shift = input - input.max(axis=-1, keepdims=True)
        exp_input = np.exp(input_shift)
        prob = exp_input / (exp_input.sum(-1, keepdims=True))
        prob_clip = np.clip(prob, 1e-12, 1 - 1e-12)
        loss = prob_clip - target
        batched_loss = loss / batch_size
        return batched_loss
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START
        #! Refference: https://www.zhihu.com/question/47746939 &
        #! https://programmathically.com/understanding-hinge-loss-and-the-svm-cost-function/
        #* 参考助教给的文档，此处 margin 就是 delta，默认为 5
        #! TODO
        # np.maxium 和 np.max 的区别？
        h_k = np.maximum(input + (self.margin - (input * target).sum(-1, keepdims=True)), 0)
        E_n = h_k.sum(axis=-1) - self.margin
        E = np.mean(E_n)
        return E
        # TODO END

    def backward(self, input, target):
        # TODO START
        #! TODO
        #* 所谓的合页函数，所以看着和 ReLU 是相似的
        batch_size = input.shape[0]
        marginal_grad = (self.margin - (input * target).sum(-1, keepdims=True) + input) > 0
        #* here we get a array of bool, so we need to convert it to float
        marginal_grad = marginal_grad.astype(float)
        marginal_grad -= target * np.sum(marginal_grad, -1, keepdims=True)
        loss = marginal_grad / batch_size
        return loss
        # TODO END