# Requirements

## Implement

1. `layers.py`: you should implement Layer , Linear , Relu , Sigmoid , Gelu loss.py : you should implement EuclideanLoss , SoftmaxCrossEntropyLoss , HingeLoss load_data.py : load mnist dataset
2. `utils.py`: some utility functions
3. `network.py`: network class which can be utilized when defining network architecture and
   performing training
4. `solve_net.py`: train_net and test_net functions to help training and testing (running
   forward, backward, weights update and logging information)
5. `run_mlp.py`: the main script for running the whole program. It demonstrates how to simply
   define a neural network by sequentially adding layers

If you implement layers correctly, just by running run_mlp.py , you can obtain lines of logging

information and reach a relatively good test accuracy. All the above files are encouraged to be

modified to meet personal needs.

If you implement layers correctly, just by running run_mlp.py , you can obtain lines of logging

information and reach a relatively good test accuracy. All the above files are encouraged to be

modified to meet personal needs.

## Report

1. Plot the loss and accuracy curves on the training set and the test set for all experiments.

2. Construct a neural network with one hidden layer, and compare the difference of results when using different activation functions ( Relu / Sigmoid / Gelu , at least 3 experiments needed and remember controlling variables.), as well as using different loss functions ( EuclideanLoss , SoftmaxCrossEntropyLoss , HingeLoss , at least 2 more experiments needed and remember controlling variables.) **You can discuss the difference from theperspectives of training time, convergence and accuracy.** 

3. "One hidden layer" means that besides the input neurons and output neurons, there are also one layer of hidden neurons.

4. Conducting the same experiments above, but with two hidden layers. Also, compare the

   difference of results between one-layer structure and two-layer structure. (At least 2 more

   experiments needed.)

## Bonus

1. Conduct experiments on a neural network with two hidden layers. Compare the difference of

results between one-layer structure and two-layer structure.

2. Tune the hyper-parameters such as the learning rate, the batch size, etc. Analyze how hyper

parameters influence the performance of the MLP. **NOTE**: The analysis is important. You will

not get any bonus points if you only conduct extensive experiments but ignore the analysis.

3. Consider the stability of computation when implementing the activation functions and the

loss functions.

## Note

1. The current hyperparameter settings may not be optimal for good classification
   performance. Try to adjust them to make test accuracy as high as possible.
2. Any deep learning framework or any other open source codes are NOT permitted in this
   homework. If you use them, you won't get any score from the homework.
3. The accuracy of your best model is required to exceed 95%.If not,TAs believe there must
   be something wrong with your code. Nevertheless, TAs will still go through your code for any
   possible bugs even if you reach the requirement.

# 实验设计

## 实验接口

### 基础实验

基础实验配置如下：

```python
config = {
    'learning_rate': 1e-2,
    'weight_decay': 0,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 50,
    'disp_freq': 50,
    'test_epoch': 1
}
```

基础实验接口不用太复杂，就三个函数，也即零层，单层和双层。固定线性层维度接口，也即：

1. 双层维度

```python
model = Network()
model.add(Linear('fc1', 784, 100, 0.01))
model.add(Gelu("Sigmoid"))
model.add(Linear('fc2', 100, 100, 0.01))
model.add(Relu("test"))
model.add(Linear('fc2', 100, 10, 0.01))
loss = HingeLoss(name='loss')
```

2. 单层维度

```python
model = Network()
model.add(Linear('fc1', 784, 100, 0.01))
model.add(Gelu("Sigmoid"))
model.add(Linear('fc2', 100, 10, 0.01))
loss = HingeLoss(name='loss')
```

3. 零层维度

```python
model = Network()
model.add(Linear('fc1', 784, 10, 0.01))
loss = HingeLoss(name='loss')
```

注意，零隐藏层不用带激活函数了。

### 大规模对比实验

预计需要调的参数有 `lr`，`wd`，`bs`，由于收敛很快，也没有过拟合，故而控制 `epoch` 为 50 即可。

`lr` 和 `wd` 需要做大规模对比实验，具体的参数待定。

## 实验目标

训练时间，收敛速率，最终效果。

## 零隐藏层实验

用三个损失函数分别跑，并且用 wandb 作图。

## 单隐藏层

三个激活函数固定用 `EuclideanLoss` 跑对比；三个损失函数固定用 `Relu` 跑对比。

## 双隐藏层

基础实验同单隐藏层。

## 扩展实验

1. 比较单双层结果。
2. 阐释超参数的影响。
3. 计算稳定性。
