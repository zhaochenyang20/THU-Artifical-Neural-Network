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

# #! Loss 层不是就到头了吗，forward 给谁，从谁那儿 backard
# class SoftmaxCrossEntropyLoss(object):
#     #! Softmax is a way to recurrence, and cross-entropy is the way conjuated to compute loss
#     def __init__(self, name):
#         self.name = name

#     def forward(self, input, target):
#         # TODO START
#         #! reference: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
#         #* 计算 softmax 前先减去最后一维的最大值
#         input_shift = input - input.max(axis=-1, keepdims=True)
#         exp_input = np.exp(input_shift)
#         prob = exp_input / (exp_input.sum(axis=-1, keepdims=True))
#         #* 为了避免梯度出现 NaN，需要做 clip
#         prob_clip = prob.clip(min=1e-12, max=1 - 1e-12)
#         #* 注意此处有了多个 batch，故而需要对 batch 内部的 loss 求均值
#         #! TODO? 真的要求均值吗？
#         corss_entropy = -(np.log(prob_clip) * target).sum(-1)
#         batch_loss = np.mean(corss_entropy)
#         return batch_loss
#         # TODO END


#     def backward(self, input, target):
#         # TODO START
#         #! TODO
#         input_shift = input - input.max(axis=-1, keepdims=True)
#         exp_input = np.exp(input_shift)
#         prob = exp_input / (exp_input.sum(-1, keepdims=True))
#         prprob_clip = np.clip(prob, 1e-12, 1 - 1e-12)
#         loss = -(target / prprob_clip)
#         batched_loss = loss / input.shape[0]
#         return batched_loss
#         # TODO END

class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        exp_input = np.exp(input - input.max(-1, keepdims=True))
        prob = exp_input / (exp_input.sum(-1, keepdims=True))
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        return (-np.log(prob) * target).sum(-1).mean()
        # TODO END

    def backward(self, input, target):
        # TODO START
        exp_input = np.exp(input - input.max(-1, keepdims=True))
        prob = exp_input / (exp_input.sum(-1, keepdims=True))
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        return (prob - target) / input.shape[0]
        # TODO END

# class HingeLoss(object):
#     def __init__(self, name, margin=5):
#         self.name = name
#         self.margin = margin

#     #! TODO
#     def forward(self, input, target):
#         # TODO START
#         #! Refference: https://www.zhihu.com/question/47746939 &
#         #! https://programmathically.com/understanding-hinge-loss-and-the-svm-cost-function/
#         #* 参考助教给的文档，此处 margin 就是 delta，默认为 5
#         #! TODO
#         h_k = np.max((self.margin - (input * target).sum(-1, keepdims=True) + input), 0)
#         E_n = h_k.sum(axis=-1) - self.margin
#         E = np.mean(E_n)
#         return E
#         # TODO END

#     def backward(self, input, target):
#         # TODO START
#         #! TODO
#         #* 所谓的合页函数，所以看着和 ReLU 是相似的
#         batch_size = input.shape[0]
#         marginal_grad = (self.margin - (input * target).sum(-1, keepdims=True) + input) > 0
#         #* here we get a array of bool, so we need to convert it to float
#         marginal_grad = marginal_grad.astype(float)
#         marginal_grad -= target * np.sum(marginal_grad, -1, keepdims=True)
#         loss = marginal_grad / batch_size
#         return loss
#         # TODO END

class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START
        return (np.maximum((self.margin - (input * target).sum(-1, keepdims=True) + input), 0).sum(-1) - self.margin).mean()
        # TODO END

    def backward(self, input, target):
        # TODO START
        grad = ((self.margin - (input * target).sum(-1, keepdims=True) + input) > 0).astype(float)
        grad -= target * np.sum(grad, -1, keepdims=True)
        return grad / input.shape[0]
        # TODO END