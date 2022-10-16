# ANN PA2 基础部分

赵晨阳 2020012363 计 06

## How `self.training` Work

对于学习任务而言，显然训练逻辑与推理逻辑是不同。例如训练时可能会根据 loss 计算梯度，而后回传梯度，参数下降；然而做推理时，大多不需要再修改参数。又比如，`BatchNormalize` 方法需要在训练时利用累计的方法来近似全局的 `average` 和 `variance`，而推理时，这两个参数需要固定，不能再变动。因此，区分训练与否是非常重要的。

具体到 `torch` 而言，`torch.nn,Module` 以及其子类都有成员变量 `self.training`，加以显示地区分是否为训练。此外，还有两个成员函数 `self.train()` 和 `self.eval()` 作为 hook，将模型的各个部分的 `self.training` 设置为 `True / False`。

## 训练效果

### MLP

搜索的参数空间如下：

```python
    batch_sizes = [512, 1024, 2048, 4096]
    learning_rates = [0.001, 0.0005, 0.0001]
    drop_rates = [0, 0.2, 0.4, 0.6, 0.8]
```

下图按照 `val_loss` 降序排序，展示了 MLP 的整体训练效果。注意 60 个完整的实验只展示了 `val_loss` 最高的 10 条。每条曲线命名规则为 `{batch_size}_{learning_rate}_{dropout_rate}`

<div align=center>
<img width="600" src="./pics/MLP_train_acc.png"/>
</div>
<div align=center>MLP Training Accuracy<br/></div>

<div align=center>
<img width="600" src="./pics/MLP_train_loss.png"/>
</div>
<div align=center>MLP Training Loss<br/></div>

<div align=center>
<img width="600" src="./pics/MLP_val_acc.png"/>
</div>
<div align=center>MLP Validation Accuracy<br/></div>

<div align=center>
<img width="600" src="./pics/MLP_val_loss.png"/>
</div>
<div align=center>MLP Validation Loss<br/></div>

如图可见，最优的曲线为 `1024_0.001_0.2`，也即 `batch_size = 1024`，`learning_rate = 0.001`，`dropout_rate = 0.2`。

### CNN

搜索的参数空间如下：

```python
    batch_sizes = [128, 256, 512, 1024]
    learning_rates = [0.001, 0.0005, 0.0001]
    drop_rates = [0, 0.2, 0.4, 0.6, 0.8]
```

下图按照 `val_loss` 降序排序，展示了 CNN 的整体训练效果。注意 60 个完整的实验只展示了 `val_loss` 最高的 10 条。每条曲线命名规则为 `{batch_size}_{learning_rate}_{dropout_rate}`

<div align=center>
<img width="600" src="./pics/CNN_train_acc.png"/>
</div>
<div align=center>CNN Training Accuracy<br/></div>

<div align=center>
<img width="600" src="./pics/CNN_train_loss.png"/>
</div>
<div align=center>CNN Training Loss<br/></div>

<div align=center>
<img width="600" src="./pics/CNN_val_acc.png"/>
</div>
<div align=center>CNN Validation Accuracy<br/></div>

<div align=center>
<img width="600" src="./pics/CNN_val_loss.png"/>
</div>
<div align=center>CNN Validation Loss<br/></div>

如图可见，最优的曲线为 `128_0.001_0.2`，也即 `batch_size = 128`，`learning_rate = 0.001`，`dropout_rate = 0.2`。

### 模型对比

CNN 模型的最高 `val_acc` 为 0.7366，MLP 模型为 0.5197；CNN 模型的最低 `val_loss` 为 0.7926，而 MLP 模型为 1.400。由此见得，CNN 显著优于 MLP。

实际上，不考虑视觉的先验经验时， CNN 仅仅是共享了参数的 MLP，计算复杂度更高，而参数更少，相对而言表达力似乎更低。然而，在视觉相关问题上，CNN 有着强大的视觉先验经验。在视觉任务上的效果远超 MLP。

实际上，CNN 模型模拟了人类的视觉直觉，人眼对于图像的理解便是从微观到全局，从高频到低频，且全过程保持着二维（乃至三维）信息的传导。通过从 low-level 到 high-level 的 feature 传递，CNN 高度利用了人眼对于图像的理解机制，可以充分利用图像上相近像素间的关系，而像素间的相互关系在 self attention 当中有着更深入的展现。MLP 将高维的图像展开为一维，导致其显式丢失了图像高维的相互信息。此外，CNN 模型通过参数共享和 pooling 机制，使其具有一定的平移不变性，也提高了 CNN 在视觉任务上的泛化能力。

总归，模型的表现有模型的容量和任务适应性共同决定。在视觉任务上，CNN 惊艳的效果众所周知，在本次实验中训练效果强于 MLP 不足为奇。

