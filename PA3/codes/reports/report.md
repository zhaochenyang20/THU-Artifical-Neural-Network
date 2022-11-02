# Transformer-Based Text Generation

原始框架中，当 validation set 的 loss 超过 best validation loss 后，训练会立刻被中断。我修改了这一策略，当连续三次 validation set 的 loss 超过 best validation loss 后，终止训练。这样的提前终止方法一定程度上防止了过拟合。

相应的训练结果如下：

## Scratch

TODO：两个 Scratch 的训练结果

TODO：两个模型的测试结果

## Finetune

TODO：两个 Finetune 的训练结果

TODO：两个模型的测试结果

## 结论 1

可以见得：

1. 从 perplexity 而言：finetune 模型在 validation set 和 test set 上的 perplexity 都显著低于 scratch 模型，可以见得 pretrain + finetune 的策略在 perplexity 上具有一定优势；
2. 在 training set 上，scratch 与 finetune 的收敛效果相近。我们可以认为，在对训练集的拟合任务上，transformer 本身的能力已经很强劲了（这一点在 TODO 详细讨论）。不过，在 test set 上的差异还是说明了预训练本身能够提高模型的泛化能力；
3. 在基本参数下，模型的 BLEU 指标几乎没有区别，我个人认为这体现了 transformer 本身的强大，然而也反映了 BLEU 指标对于泛化性的反馈是不到位的；

具体而言，BLEU 比较候选译文和参考译文里的 n-gram（实践中从 unigram 取到 4-gram） 的重合程度，重合程度越高就认为译文质量越高。选不同长度的 n-gram 是因为，unigram 的准确率可以用于衡量单词翻译的准确性，更高阶的 n-gram 的准确率可以用来衡量句子的流畅性。

这是一个**只看中准确率**的指标，更关心候选译文里的多少 n-gram 是正确的（即在参考译文里也出现），而不在乎召回率（参考译文里有哪些 n-gram 在候选译文中并未出现）。不过这不算特别严重的问题，因为 BLEU 原论文**建议大家的测试集里给每个句子配备 4 条参考译文**，这样就可以减小语言多样性带来的影响（然而很多机器翻译的测试集都是只有 1 条译文）；另外还有 brevity penalty 来惩罚候选译文过短的情况（候选译文过短在机器翻译中往往意味着漏翻，也就是低召回率）。

但总的来说，学界普遍认为 **BLEU 指标偏向于较短的翻译结果**，因为 brevity penalty 没有想象中那么强。

总而言之，BLEU 具有如下缺点：

1. 　不考虑语言表达（语法）上的准确性；

2. 　测评精度会受常用词的干扰；

3. 　短译句的测评精度有时会较高；

4. 　没有考虑同义词或相似表达的情况，可能会导致合理翻译被否定；

# Decoding Strategies

| 模型 / 指标   | val_ppl | val_loss | Train_loss |
| ------------- | ------- | -------- | ---------- |
| 3_128         | 24.095  | 3.228    | 2.112      |
| 12_64         | 25.069  | 3.23     | 1.843      |
| primary_bs128 | 19.604  | 2.829    | 1.968      |
| full_bs58     | 15.454  | 2.829    | 1.945      |

```python
experiments = [
    ("random", 1, 0, 0),
    ("random", 0.7, 0, 0),
    ("top-p", 1, 0.9, 0),
    ("top-p", 0.7, 0.9, 0),
    ("top-k", 1, 0, 40),
    ("top-k", 0.7, 0, 40),
]
# inference 策略，temperature，top-p，top-k
```

1. 对于 temperature 参数：在三种策略下，temperature 参数减小都会导致 fw-bleu-4 增大，bw-bleu-4 减小，反映出 temprature 减小会加强单个句子的质量，但是会降低句子的多样性。实际上，temperature 参数作用与解码器的输出层。输出层后通常会通过 softmax 函数来将输出概率归一化，通过改变 temperature 可以控制概率的形貌。当 temperature 较大的时候，概率分布趋向平均，随机性增大；当 temperature 较小的时候，概率密度趋向于分散，高概率者出现的可能更强，随机性降低。

$$
p_i=\frac{\exp \left(y_i / T\right)}{\sum \exp \left(y_i / T\right)}
$$

2. 对于 p 参数，在 top-p decoding 时，p 减小都会导致 fw-bleu-4 增大，bw-bleu-4 减小，反映出 p 减小会加强单个句子的质量，但是会降低句子的多样性。具体而言，Top-p (nucleus) sampling 是 Ari Holtzman et al. (2019) 提出的算法。从使得累计概率超过 p 的最小候选集里选择单词，然后算这些单词的概率分布。这样候选单词集的大小会随下一个单词的概率分布动态增加和减少。当 p 减小时，模型筛选出更少的单词集合，提升了单一句子质量，然而选项更少，句子的丰富度降低；反之，p 增大时，模型筛选出更多的单词集合，可能会降低单一句子质量，然而选项更多，能产生更加丰富的句子；当 p = 1.0 时，实际上已经等同于 random 策略。另外，在实际研究中，往往会为 top-p 策略加入 minimal candidate 参数，避免候选的单词集大小过小。

3. 对于 k 参数，同样，p 减小都会导致 fw-bleu-4 增大，bw-bleu-4 减小，反映出 k 减小会加强单个句子的质量，但是会降低句子的多样性。实际上，k 直接限定了单词集合的大小。当 k 减小时，模型筛选出更少的单词集合，提升了单一句子质量，然而选项更少，句子的丰富度降低；反之，k 增大时，模型筛选出更多的单词集合，可能会降低单一句子质量，然而选项更多，能产生更加丰富的句子。

```python
temperatures = [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6]
model = "./train_ckpt/3_128.tar"
```

