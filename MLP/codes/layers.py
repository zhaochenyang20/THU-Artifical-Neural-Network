import numpy as np
from sympy import re


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    # ! acutally we won't use the base class Layer, so these function are not required
    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Relu(Layer):
    #! 什么是激活函数层，为什么要在此存下 _saved_for_backward，而且都是 input 而非激活后的 activation
    # 注意在计算梯度时，需要存下当前层的 input（求导是计算当前输入下的导数），然而梯度是递乘的，从后一直传递向前，故而需要乘上 grad_output
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        # TODO START
        activatd_input = np.maximum(0, input)
        # 对于 Relu 而言，不能存下激活后的 activation，而是存下 input，否则在 backward 时，梯度会断掉
        self._saved_for_backward(input)
        return activatd_input
        # TODO END

    def backward(self, grad_output):
        # TODO START
        #* ReLU 导函数为 1 if x > 0 else 0
        if self._saved_tensor is None:
            raise ValueError('No saved tensor for backward')
        elif self._saved_tensor is not None:

            def diriviation_Relu(x):
                return 1 if x > 0 else 0

            grad_backword = grad_output * (self._saved_tensor > 0)

            return grad_backword
        # TODO END

class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        # TODO START

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        activatd_input = sigmoid(input)
        self._saved_for_backward(input)
        return activatd_input

        # TODO END

    def backward(self, grad_output):
        # TODO START

        def diriviation_sigmoid(x):
            return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

        grad_backword = grad_output * diriviation_sigmoid(self._saved_tensor)
        return grad_backword

        # TODO END


class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)

    def forward(self, input):
        # TODO START
        def gelu(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

        activatd_input = gelu(input)
        self._saved_for_backward(input)
        return activatd_input
        # TODO END

    def backward(self, grad_output):
        # TODO START
        #! references: https://alaaalatif.github.io/2019-04-11-gelu/
        #! references: for the inplimentation of gelu's derviation, I asked for help from Zhiyuan Zeng, StudentID 2020010864
        #* I suspect that TAs wanna us to use delta = 1e(-5) to compute the derivatives

        def diriviation_gelu(x):
            return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) \
                * (x + 0.044715 * np.power(x, 3)))) + 0.5 * x * np.sqrt(2 / np.pi) * \
                    (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))\
                        ** 2 * (1 + 3 * 0.044715 * np.power(x, 2))

        def apropriate_derivative_gelu(x):
            def gelu(x):
                return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
            delta = 1e-6
            return (gelu(x + delta) - gelu(x - delta)) / (2 * delta)

        grad_backword = grad_output * apropriate_derivative_gelu(self._saved_tensor)
        return grad_backword
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        #! TODO add readme
        import numpy as np
        np.random.seed(1)
        #! add readme
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

        #* grad 是一般意义上的梯度，而 diff 是在 Adam 当中利用的冲量

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        matmul_result = np.matmul(input, self.W)
        forward = matmul_result + self.b
        return forward
        # TODO END

    def backward(self, grad_output):
        # TODO START
        if self._saved_tensor is None:
            raise ValueError('No saved tensor for backward')
        else:
            self.grad_W = np.matmul(self._saved_tensor.T, grad_output)
            self.grad_b = grad_output.sum(0)
            backward = np.matmul(grad_output, self.W.T)
            return backward
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
