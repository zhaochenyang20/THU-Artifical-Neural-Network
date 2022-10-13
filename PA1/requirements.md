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

1. Conduct experiments on a neural network with two hidden layers. Compare the difference of results between one-layer structure and two-layer structure.

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

- 比较学习率

```python
python3 run_mlp.py -n 2 -a Sigmoid -l SoftmaxCrossEntropyLoss

config = {
    'learning_rate': 1,
    'weight_decay': 0,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 50,
    'disp_freq': 50,
    'test_epoch': 1
}

python3 run_mlp.py -n 1 -a Sigmoid -l EuclideanLoss

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

- 可以用来比较 `weight_decay`

```python
python3 run_mlp.py -n 2 -a Sigmoid -l SoftmaxCrossEntropyLoss
config = {
    'learning_rate': 1,
    'weight_decay': 1e-5,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 50,
    'disp_freq': 50,
    'test_epoch': 1
}

python3 run_mlp.py -n 2 -a Gelu -l HingeLoss
config = {
    'learning_rate': 1e-2,
    'weight_decay': 1e-1,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 50,
    'disp_freq': 50,
    'test_epoch': 1
}
```


- 批量大小

```python
python3 run_mlp.py -n 2 -a Sigmoid -l SoftmaxCrossEntropyLoss
python3 run_mlp.py -n 1 -a Sigmoid -l EuclideanLoss
```

\documentclass{article}

\usepackage{arxiv}
\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{natbib}
\usepackage{doi}
\usepackage{ctex}
\usepackage{multirow}
\usepackage{booktabs}

\title{人工神经网络 MLP Numpy}
\date{\today}
\author{
	赵晨阳\\
	2020012363\\
	清华大学计算机系\\
	\texttt{zhaochenyang20@gmail.com}
}

\renewcommand{\headeright}{人工神经网络}
\renewcommand{\undertitle}{}
\renewcommand{\shorttitle}{MLP Numpy}

\begin{document}
\maketitle

\section{基本实验}

\subsection{参数设定}
基础实验部分采用的超参数如下所示，注意到部分网络架构在该超参数设定下可能无法成功训练。
\begin{verbatim}
	config = {
    'learning_rate': 1e-2,
    'weight_decay': 0,
    'momentum': 0.0,
    'batch_size': 10,
    'max_epoch': 50,
    'disp_freq': 50,
    'test_epoch': 1
}
\end{verbatim}

\subsection{图例说明}
如图 ~\ref{fig:1} 所示，实验结果图表标题分别展示了 test\_loss 与 train\_loss。\\
每条图例曲线命名规则为：\\
\textbf{\{hidden layer num\}\_\{activation function name\}\_\{loss function name\}\_\{learning rate\}\_\{weight decay\}\_\{batch size\}}\\
倘若没有尾缀的三个超参数，则表示采用默认参数设定。

\begin{figure}
	\centering
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层实验-test_loss.png}
		\caption{单隐藏层实验 test loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/消减率_2_Gelu_HingeLoss_train_loss.png}
		\caption{单隐藏层实验 test accuracy}
	\end{subfigure}
\caption{图例说明}
\label{fig:1}
\end{figure}

\subsection{零隐藏层实验}
对于零隐藏层网络，分别使用三种损失函数，训练过程和训练结果如图 ~\ref{fig:2} 所示。可以发现如果只有一层线性层的话，无法达到预期的学习效果。

\begin{figure}
	\centering
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/零层实验-test-loss.png}
		\caption{零隐藏层 test loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/零层实验-train-loss.png}
		\caption{零隐藏层 train loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/零层实验-test-acc.png}
		\caption{零隐藏层 test accuracy}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/零层实验-train_acc.png}
		\caption{零隐藏层 train accuracy}
	\end{subfigure}
	\caption{零隐藏层实验效果}
	\label{fig:2}
\end{figure}

\subsection{单隐藏层实验}
增加单层隐藏层，隐藏层神经元数设置为 100，遍历三种损失函数和三种激活函数进行对比实验。其结果如图 \ref{fig:3} 所示。

\begin{figure}
	\centering
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层实验-test_loss.png}
		\caption{单隐藏层 test loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层实验-train_loss.png}
		\caption{单隐藏层 train loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层实验-test_acc.png}
		\caption{单隐藏层 test accuracy}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层实验-train_acc.png}
		\caption{单隐藏层 train accuracy}
	\end{subfigure}
	\caption{单隐藏层实验效果}
	\label{fig:3}
\end{figure}

\subsubsection{不同损失函数函数对比}

在单隐藏层的前提下，固定激活函数为 Sigmoid，遍历三种损失函数，结果如图 \ref{fig:4} 所示。\\
分析结果可以发现，（1）就准确率而言，在测试集和训练集上均是 Hinge 效果最好而 Euclidean 效果最差；（2）就损失值而言，Hinge、SoftmaxCrossEntropy 和 Euclidean 计算出的绝对值存在着明显的差异，损失值无法横向比较，然而对比损失值下降速度，能够观察到 Hinge 收敛效果与收敛速度最为显著；（3）就收敛速度而言，Hinge 收敛曲线斜率最高，收敛最快，而 Euclidean 收敛最慢；（4）就训练时间而言，Hinge 耗时为 
10m 49s，Euclidean 耗时为 
10m 45s，SoftmaxCrossEntropy 耗时也为 10m 45s，三者并无显著差异。
\\
比较 Hinge、SoftmaxCrossEntropy 和 Euclidean 的函数形式可见，Hinge 损失函数的计算形式最为复杂，效果出人意料更好。然而之前接触过的分类任务更多使用 SoftmaxCrossEntropy。没有选择使用 Hinge 的原因可能是 Hinge 还还存在超参数 Margin 的缘故。对于 Margin 参数的 tune 也能作为实验目标。

\begin{figure}
	\centering
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层损失函数test_loss.png}
		\caption{单隐藏层损失函数对比 test loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层损失函数train_loss.png}
		\caption{单隐藏层损失函数对比 train loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层损失函数test_acc.png}
		\caption{单隐藏层损失函数对比 test accuracy}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层损失函数train_acc.png}
		\caption{单隐藏层损失函数对比 train accuracy}
	\end{subfigure}
	\caption{单隐藏层对比损失函数}
	\label{fig:4}
\end{figure}

\subsubsection{不同激活函数对比}

在单隐藏层的前提下，固定损失函数为 Euclidean，遍历三种激活函数，结果如图 \ref{fig:5} 所示。


分析结果可以发现，（1）就准确率而言，ReLU 结果略微高于 GeLU，而 Sigmoid 的表现与二者有一定差距；（2）就损失值而言，ReLU 损失值略微低于 GeLU，而 Sigmoid 损失值显著大于前两者，且损失值衰减速率基本与准确率的上升速率符合；（3）就收敛速度而言，ReLU 与 Gelu 收敛速度相当，而 Sigmoid 收敛最慢，且收敛值最低；（4）就训练时间而言，Sigmoid 耗时为
 10m 45s，Relu 为 10m 33s，Gelu 为 
11m 14s，考虑到 Relu 最为简单的函数形式，具有最快的训练速度符合预期。而 Gelu 的实现方式基于微分的极限定义，计算复杂度最高，也符合预期。\\
对比图 \ref{fig:5b}，考虑到三者的函数定义。
\begin{itemize}
    \item $Gelu(x)=x \times P(X<=x)=x \times \phi(x), x \sim N(0,1) \approx  0.5 \times x\left(1+\tanh \left[\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^3\right)\right]\right)$
    \item $Relu(x)=x^{+}=\max (0, x)$
    \item $Sigmoid(x)=\frac{1}{1+e^{-x}}=\frac{e^x}{e^x+1}=1-Sigmoid(-x)$
\end{itemize}
\\Sigmoid 和 GeLU 光滑且连续可微的函数；ReLU 与 Gelu 图像和导数值相近，在损失值为正时，具有相对稳定的导数。然而，Sigmoid 的效果一贯较差是深度学习领域研究者多年得出的一贯结论。具体而言，Sigmoid 在损失值较大时，实际上倒数值偏小，梯度消失非常严重。而我们观察到对 Euclidean 损失而言，初始的损失值就是偏大的，导致 Sigmoid 一直处于倒数值偏小的区间，严重限制了收敛效率与最终的收敛效果。

\begin{figure}
	\centering
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层激活函数test_loss.png}
		\caption{单隐藏层激活函数对比 test loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层激活函数train_loss.png}
		\caption{单隐藏层激活函数对比 train loss}
		\label{fig:5b}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层激活函数test_acc.png}
		\caption{单隐藏层激活函数对比 test accuracy}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/单层激活函数train_acc.png}
		\caption{单隐藏层激活函数对比 train accuracy}
	\end{subfigure}
	\caption{单隐藏层对比激活函数}
	\label{fig:5}
\end{figure}

\subsubsection{过拟合问题}

实际上在本次实验中并未出现明显的过拟合现象，基本上实验结果保证了 loss 和 accuracy 负相关。然而，这一现象实际上并不绝对，在复杂模型处理复杂任务的情况下，往往会出现严重的过拟合。

\subsection{双隐藏层实验}

增加双层隐藏层，隐藏层神经元数分别设置为 100 和 100，遍历不同损失函数和激活函数进行实验，得到的训练过程和训练结果分别如图~\ref{fig:6}所示。相比同样单隐藏层训练结果，双隐藏层的网络在测试集、训练集上的表现并未与单隐藏层有显著优势。甚至在许多模型设定下不如单隐藏层。出于常识，双隐藏层网络比单隐藏层网络游拥有更多的参数，似乎该具有更强的学习能力。
\\然而，这个结论并不必然。一方面，知识蒸馏领域便希望能够用更小的模型达到更好的效果。另一方面，在这次实验中，双层的参数复杂度上升，然而没有加入 dropout 机制，也没有设置 residual connection。考虑到 MNIST 十分类
这个任务比较简单，所以1个隐藏层的FFN模型表达能力已经足够强在较为简单的十分类任务下，结果不如单隐藏层，也符合常理。最后，我没有尝试调整 momentum 参数，等效于使用了最为原始的 SGD optimizer，也没有体现出增加隐藏层的优势。
\\此外，注意到实际上双隐藏层时，某些模型设定并不能成功训练。考虑到训练失败的设定是 2\_Sigmoid\_SoftmaxCrossEntropy 和 2\_Sigmoid\_Euclidean，2\_Sigmoid\_Hinge 成功训练；叠加双层的 Sigmoid 作为激活函数，可能会存在非常严重的梯度消失问题，只有 Hinge 能够得到训练，这也符合预期。
\\当然，这里所指的不能成功训练是基于基础实验的超参数设计而言的，实际上讲学习率从 1e-2 提高到 1 即可让 2\_Sigmoid\_SoftmaxCrossEntropy 成功训练；这也符合预期，因为提高学习率也是对抗梯度衰减的一大方法。实际上，扩展实验中对学习率的调节便是基于 2\_Sigmoid\_SoftmaxCrossEntropy 而展开。而 2\_Sigmoid\_Euclidean 设定及时调整了学习率，依然难以训练成功。据此推断，没有使用 residual connection 的情况下，层数越多 Sigmoid 导致的梯度消失问题越严重，劣势进一步被放大。

\begin{figure}
	\centering
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/双层实验-test_loss.png}
		\caption{双隐藏层 test loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/双层实验-train_loss.png}
		\caption{双隐藏层 train loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/双层实验-test_acc.png}
		\caption{双隐藏层 test accuracy}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/双层实验-train_acc.png}
		\caption{双隐藏层 train accuracy}
	\end{subfigure}
	\caption{双隐藏层实验效果}
	\label{fig:6}
\end{figure}

\section{扩展实验}

\subsection{学习率影响}

保留基础实验的参数设定，模型选取 1\_Sigmoid\_EuclideanLoss（图 \ref{fig:7}） 与 2\_Sigmoid\_Softmax（图 \ref{fig:8}），学习率选取为 \{1e-4, 1e-3, 1e-2, 1e-1, 1\}，展开学习率对比实验。

\begin{figure}
	\centering
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/学习率_1_Sigmoid_EuclideanLoss_test_loss.png}
		\caption{1\_Sigmoid\_EuclideanLoss 对比 test loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/学习率_1_Sigmoid_EuclideanLoss_train_loss.png}
		\caption{1\_Sigmoid\_EuclideanLoss 对比 train loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/学习率_1_Sigmoid_EuclideanLoss_test_acc.png}
		\caption{1\_Sigmoid\_EuclideanLoss 对比 test accuracy}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/学习率_1_Sigmoid_EuclideanLoss_train_acc.png}
		\caption{1\_Sigmoid\_EuclideanLoss 对比 train accuracy}
	\end{subfigure}
	\caption{基于 1\_Sigmoid\_EuclideanLoss 调节学习率}
	\label{fig:7}
\end{figure}

\begin{figure}
	\centering
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/学习率_2_Sigmoid_Softmax_test_loss.png}
		\caption{2\_Sigmoid\_Softmax 对比 test loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/学习率_2_Sigmoid_Softmax_train_loss.png}
		\caption{2\_Sigmoid\_Softmax 对比 train loss}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/学习率_2_Sigmoid_Softmax_test_acc.png}
		\caption{2\_Sigmoid\_Softmax 对比 test accuracy}
	\end{subfigure}
	\begin{subfigure}{0.475\textwidth}
		\centering
		\includegraphics[width=1\textwidth]{../pics/学习率_2_Sigmoid_Softmax_train_acc.png}
		\caption{2\_Sigmoid\_Softmax 对比 train accuracy}
	\end{subfigure}
	\caption{基于 2\_Sigmoid\_Softmax 调节学习率}
	\label{fig:8}
\end{figure}

基于深度学习领域的长期共识，学习率的调整需要基于对于模型本身结构的认知，并没有普适的优秀学习率能够提供普遍优秀的学习效果；譬如大多数模型适用的 1e-2 学习率在 2\_Sigmoid\_Softmax 和 2\_Sigmoid\_Euclidean 完全失败。总体上，对于梯度消失非常严重的双层 Sigmoid 激活函数模型，需要采用较大的学习率，而基于其他激活函数的大多模型能够依靠 1e-2 成功训练。
\\
对于 1\_Sigmoid\_EuclideanLoss 而言可以发现，随着学习率的减小模型在测试集上的表现会显著降低。此外，由于本次实验任务较为简单的缘故，此处并没有出现过拟合的情况。观察训练过程的曲线可以发现，随着学习率的增大，参数更新的幅度也增大，模型的收敛速度也会变快，模型的表现会出现较大的变动。可以猜测，大于 1 后，更大的学习率可能会导致模型难以接近最优的收敛解，最终模型在测试集上的表现也会降低（这一情况并未出现在此次实验中）；而较小的学习率却会导致收敛速度过慢以至于在设定的 epoch 次数上训练难以达到最优解，最终模型在测试集上的表现也会降低。

\subsection{批量大小影响}

使用GeLU作为激活函数，Euclidean作为损失函数，调整批量大小（batch size）分别为50、100和200，得到的训练过程和训练结果分别如图~\ref{fig:batch_size}和表~\ref{tab:batch_size}所示。



可以发现，在增大或者减小批量大小时，模型在测试集上的表现都会降低。但是随着批量大小的增加，模型在测试集上的波动却会变小。具体原因可能是批量大小与每次梯度反向传播与参数更新幅度密切相关。当批量大小增加时，每次梯度反向传播数值都比较稳定，因此准确率表现的波动较小；但是每次参数更新幅度也比较小，因此也难以达到最优结果。



\end{document}
