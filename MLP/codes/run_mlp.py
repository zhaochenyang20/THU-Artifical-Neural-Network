from re import S
from network import Network
from utils import LOG_INFO
from layers import Gelu, Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Linear('fc1', 784, 100, 0.01))
model.add(Gelu("Sigmoid"))
model.add(Linear('fc2', 100, 100, 0.01))
model.add(Relu("test"))
model.add(Linear('fc2', 100, 10, 0.01))
loss = HingeLoss(name='loss')


# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 1e-2,
    'weight_decay': 0,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 50,
    'disp_freq': 50,
    'test_epoch': 1
}

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])
