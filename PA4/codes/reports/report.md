# ANN Finale —— GAN

[toc]

# 基础实验

## 实验设定

图例中的线条图例遵从如下的实验命名原则：

```python
GAN_{backbone}_latent_dim_{latent_dim}_generator_hidden_dim_{generator_hidden_dim}_discriminator_hidden_dim_{discriminator_hidden_dim}_seed_{seed}
```

其中，`backbone` 为 `{"MLP", "CNN"}`；默认 `seed` 为 42（如果 `seed` 为 42 则省略该项）

其余部分采用默认参数，如下所示：

```python
batch_size = 64
num_training_steps = 5000
learning_rate = 0.0002
beta1 = 0.5
```

为了完成本次作业，我一共进行了 18 组基础实验，其配置参数如下：

```python
latent_dims = [16, 64, 100]
hidden_dims = [16, 64, 100] 
# 将 generator_hidden_dim 与 discriminator_hidden_dim 设定为相同的大小
backbones = ["CNN", "MLP"]
experiments = product(latent_dims, hidden_dims, backbones)
```

此外，为了测试 `seed` 参数对于实验的影响，我测试了如下 4 组实验：

```python
seeds = [43, 107, 213, 996]
latent_dim = 16
generator_hidden_dim = 16
discriminator_hidden_dim = 16
backbone = "CNN"
```

所有的实验结果可以参考[此链接](https://wandb.ai/eren-zhao/GAN/overview?workspace=GAN)。

##  `latent_dim` 与 `hidden_dim`

为了更加显著地显示 `latent_dim` 与 `hidden_dim` 带给模型的差异，我选择展示如下四组实验的训练曲线与最后一次生成结果：

```python
latent_dims = [16, 100]
hidden_dims = [16, 100] 
backbones = "CNN"
```

以下小节标题为 {backbone} {latent dim} {hidden_dim}

### CNN 16 16

<div align=center>
<img width="600" src="./pics/CNN_16_16_16.png"/>
<img width="600" src="./pics/CNN_16_16_16 (2).png"/>
<img width="600" src="./pics/CNN_16_16_16 (3).png"/>
<img width="600" src="./pics/CNN_16_16_16 (1).png"/>
<img width="600" src="./pics/CNN_16_16_16 (4).png"/>
<img width="600" src="./CNN_16_16_16/4999/samples.png"/>
</div>

### CNN 16 100

<div align=center>
<img width="600" src="./pics/CNN_16_100_100.png"/>
<img width="600" src="./pics/CNN_16_100_100 (2).png"/>
<img width="600" src="./pics/CNN_16_100_100 (3).png"/>
<img width="600" src="./pics/CNN_16_100_100 (1).png"/>
<img width="600" src="./pics/CNN_16_100_100 (4).png"/>
<img width="600" src="./CNN_16_100_100/4999/samples.png"/>
</div>

### CNN 100 16

<div align=center>
<img width="600" src="./pics/CNN_100_16_16.png"/>
<img width="600" src="./pics/CNN_100_16_16 (2).png"/>
<img width="600" src="./pics/CNN_100_16_16 (3).png"/>
<img width="600" src="./pics/CNN_100_16_16 (1).png"/>
<img width="600" src="./pics/CNN_100_16_16 (4).png"/>
<img width="600" src="./CNN_100_16_16/4999/samples.png"/>
</div>

### CNN 100 100

<div align=center>
<img width="600" src="./pics/CNN_100_100_100.png"/>
<img width="600" src="./pics/CNN_100_100_100 (2).png"/>
<img width="600" src="./pics/CNN_100_100_100 (3).png"/>
<img width="600" src="./pics/CNN_100_100_100 (1).png"/>
<img width="600" src="./pics/CNN_100_100_100 (4).png"/>
<img width="600" src="./CNN_100_100_100/4999/samples.png"/>
</div>
## `FID`

| Model Config    | FID Score          |
| --------------- | ------------------ |
| CNN_16_16_16    | 104.45376654460628 |
| CNN_16_100_100  | 31.057871640232264 |
| CNN_100_16_16   | 65.38224467437774  |
| CNN_100_100_100 | 36.13799152833505  |

可以见得，四组模型中，CNN_16_100_100 模型的 FID 最佳，取得了 31.06 的优秀表现。

## 超参讨论

<img width="600" src="./pics/FID.png"/>

从上图可见，就 `FID` 指标而言，对于 MLP 和 CNN，`hidden_dim` 增大都能带来训练效果的提升。

`hidden_num` ⼏乎直接体现了模型的表⽰⼒，其增⼤导致指标的提升是合理的。在不考虑训练成本的前提下，增大 `hidden_dim` 会增强 generator 的生成质量；同时，也有利于增强 discriminator 的分类能力。

训练过程中，generator 的生成潜能得到了强化，而更强劲的 discriminator 能够迫使生成器产生更好的生成效果。无论是从 generator 还是 discriminator 的角度而言，增大 `hidden_dim` 都有助于提升模型的表现能力。

<img width="600" src="./pics/FID2.png"/>

如上图所示，而考虑 `latent_dim` 时，可以见得其参数变动对于模型的影响相对没有规律可言。考虑到 `latent_dim` 主要是⽤于随机噪声的维度，⽽过⼤的 `latent_dim` 相当于更随机的噪声，这对于⼀个⽣成器来说并不⼀定会导致更好的效果，反而可能加⼤学习难度致使指标下降。

总体来看，目前来看过大或过小的 `latent_dim` 都不合适、取一个适中的值会比较好。

## 纳什均衡

在 GAN 一文的原文中，作者提到了 GAN 模型的纳什均衡点为 0.5。这是最理想的情况下，生成器生成的图像完全可以以假乱真，辨别器随机的输出正误。

此处参考 CNN_100_100 模型的训练曲线：

<div align=center>
<img width="600" src="./pics/CNN_100_100_100.png"/>
<img width="600" src="./pics/CNN_100_100_100 (2).png"/>
</div>

训练全程的 `discriminator_loss` 和 `generator_loss` 都在剧烈波动，远远没有达到纳什均衡。实际上，台大李宏毅老师曾经幽默地形容道：“No Pain No GAN!” GAN 的训练难度出奇的高，往往需要用到如下的一些技巧：

1. 梯度惩罚
2. 优先训练 Discriminator
3. 标签平滑
4. 添加噪声
5. 不使用基于动量的优化算法，推荐 RMSProp 等等
6. 在生成器和判别器中使用 Batch Normalization

本次实验没有实现其中任一技巧，用朴素的同时训练法难以达到优良的训练效果可想而知。

