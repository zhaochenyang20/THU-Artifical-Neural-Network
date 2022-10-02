import os


def pipeline():
    """Pipeline to run the whole project."""
    act_funcs = ['Gelu', 'Relu', 'Sigmoid']
    loss_functions = ['HingeLoss', 'SoftmaxCrossEntropyLoss', 'EuclideanLoss']
    # 0 layers
    for loss_function in loss_functions:
        os.system('python3 run_mlp.py --hidden_layers_num 0 --activate_function None --loss_function ' + loss_function)
    # 1 layers
    for activate_function in act_funcs:
        for loss_function in loss_functions:
            os.system('python3 run_mlp.py --hidden_layers_num 1 --activate_function ' + activate_function + ' --loss_function ' + loss_function)
    # 2 layers
    for activate_function in act_funcs:
        for loss_function in loss_functions:
            os.system('python3 run_mlp.py --hidden_layers_num 2 --activate_function ' + activate_function + ' --loss_function ' + loss_function)

if __name__ == "__main__":
        pipeline()