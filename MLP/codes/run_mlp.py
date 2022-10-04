from network import Network
from utils import LOG_INFO
from layers import Gelu, Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d

config = {
    'learning_rate': 1e-2,
    'weight_decay': 0,
    'momentum': 0.0,
    'batch_size': 10,
    'max_epoch': 50,
    'disp_freq': 50,
    'test_epoch': 1
}

#! 双层 sigmoid，softmax 用 lr = 1，其他 0.1
def parser_data():
    import argparse
    parser = argparse.ArgumentParser(prog='Basic experiments', description='Basic experiment in MLP', allow_abbrev=True)

    parser.add_argument('-n', '--hidden_layers_num', dest='hidden_layers_num', type=int, default=None, required=True, help='The number of hidden layers in the model.')
    parser.add_argument('-a', '--activate_function', dest='activate_function', type=str, default=None, required=False, help="the activation function to be used")
    parser.add_argument('-l', '--loss_function', dest='loss_function', type=str, default=None, required=True, help="the loss function to be used")
    parser.add_argument(
        "--t",
        action="store_true",
        dest="tune",
        help="use to massive tune experiments",
    )
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=config['learning_rate'], help='the learning rate')
    parser.add_argument('-w', '--weight_decay', dest='weight_decay', type=float, default=config['weight_decay'], required=False, help='the weight decay')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=config['batch_size'], required=False, help='the batch size')

    args = parser.parse_args()
    hidden_layers_num = args.hidden_layers_num
    activate_function = args.activate_function
    loss_function = args.loss_function

    config['learning_rate'] = args.learning_rate
    config['weight_decay'] = args.weight_decay
    config['batch_size'] = args.batch_size

    tune = args.tune
    if tune:
        appendix_name = str(config["learning_rate"]) + "_" + str(config["weight_decay"]) + "_" + str(config["batch_size"])
    else:
        appendix_name = None

    return (hidden_layers_num, activate_function, loss_function, appendix_name)

def model_and_loss_generator(hidden_layers_num, activate_function, loss_function):
    model = Network()
    if hidden_layers_num == 0:
        model.add(Linear('fc1', 784, 10, 0.01))

    elif hidden_layers_num == 1:
        model.add(Linear('fc1', 784, 100, 0.01))
        if activate_function == 'Gelu':
                model.add(Gelu("Gelu"))
                model.add(Linear('fc2', 100, 10, 0.01))
        elif activate_function == 'Relu':
            model.add(Relu("Relu"))
            model.add(Linear('fc2', 100, 10, 0.01))
        elif activate_function == 'Sigmoid':
            model.add(Sigmoid("Sigmoid"))
            model.add(Linear('fc2', 100, 10, 0.01))
        else:
            raise ValueError("Unknown activation function")

    elif hidden_layers_num == 2:
        model.add(Linear('fc1', 784, 100, 0.01))
        if activate_function == 'Gelu':
            model.add(Gelu("Gelu"))
            model.add(Linear('fc1', 100, 100, 0.01))
            model.add(Gelu("Gelu"))
            model.add(Linear('fc1', 100, 10, 0.01))
        elif activate_function == 'Relu':
            model.add(Relu("Relu"))
            model.add(Linear('fc1', 100, 100, 0.01))
            model.add(Relu("Relu"))
            model.add(Linear('fc1', 100, 10, 0.01))
        elif activate_function == 'Sigmoid':
            model.add(Sigmoid("Sigmoid"))
            model.add(Linear('fc1', 100, 100, 0.01))
            model.add(Sigmoid("Sigmoid"))
            model.add(Linear('fc1', 100, 10, 0.01))
        else:
            raise ValueError("Unknown activation function")

    if loss_function == 'HingeLoss':
        loss = HingeLoss(name='loss')
    elif loss_function == 'SoftmaxCrossEntropyLoss':
        loss = SoftmaxCrossEntropyLoss(name='loss')
    elif loss_function == 'EuclideanLoss':
        loss = EuclideanLoss(name='loss')
    else:
        raise ValueError("Unknown loss function")
    return model, loss


def main(model, loss, run_name):
    import wandb
    wandb.init(project="Test New Code", name=f"{run_name}")
    print(run_name)

    train_data, test_data, train_label, test_label = load_mnist_2d('data')

    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_net(model, loss, test_data, test_label, config['batch_size'])

if __name__ == '__main__':
    hidden_layers_num, activate_function, loss_function, appendix_name = parser_data()
    model, loss = model_and_loss_generator(hidden_layers_num, activate_function, loss_function)
    if appendix_name != None:
        run_name = str(hidden_layers_num)  + (("_" + str(activate_function)) if activate_function else "") + "_" + str(loss_function) + ("_" + appendix_name)
    else:
        run_name = str(hidden_layers_num)  + (("_" + str(activate_function)) if activate_function else "") + "_" + str(loss_function)
    print(run_name)
    main(model, loss, run_name)