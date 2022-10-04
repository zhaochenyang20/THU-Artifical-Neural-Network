import os

def basic_experiment():
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


def tune():
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
    weight_decays = [0.0001, 0.001, 0.01, 0.1, 1]
    batch_sizes = [10, 20, 50, 100, 200]
    learning_rate_competers = ["python3 run_mlp.py -n 2 -a Sigmoid -l SoftmaxCrossEntropyLoss ","python3 run_mlp.py -n 1 -a Sigmoid -l EuclideanLoss "]
    weight_decay_components = ["python3 run_mlp.py -n 2 -a Sigmoid -l SoftmaxCrossEntropyLoss ", "python3 run_mlp.py -n 2 -a Gelu -l HingeLoss "]
    batch_size_components = ["python3 run_mlp.py -n 2 -a Sigmoid -l SoftmaxCrossEntropyLoss ", "python3 run_mlp.py -n 1 -a Sigmoid -l EuclideanLoss "]
    for learning_rate_competer in learning_rate_competers:
        for learning_rate in learning_rates:
            commond = learning_rate_competer + f"-lr {learning_rate} -w 0.00001 -b 100 --t"
            print(commond)
            os.system(commond)

    #! tune on weight_decay_components
    commond = weight_decay_components[0]
    for weight_decay in weight_decays:
        weight_decay_commond = commond + f"-lr 1 -w {weight_decay} -b 100 --t"
        print(weight_decay_commond)
        os.system(weight_decay_commond)
    commond = weight_decay_components[1]
    for weight_decay in weight_decays:
        weight_decay_commond = commond + f"-lr 0.01 -w {weight_decay} -b 100 --t"
        print(weight_decay_commond)
        os.system(weight_decay_commond)

    #! tune on batch_size_components
    commond = batch_size_components[0]
    for batch_size in batch_sizes:
        batch_size_commond = commond + f"-lr 1 -w 0 -b {batch_size} --t"
        print(batch_size_commond)
        os.system(batch_size_commond)
    commond = batch_size_components[1]
    for batch_size in batch_sizes:
        batch_size_commond = commond + f"-lr 0.01 -w 0 -b {batch_size} --t"
        print(batch_size_commond)
        os.system(batch_size_commond)

if __name__ == "__main__":
        # basic_experiment()
        tune()