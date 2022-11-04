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

为了方便模型的横向对比，我选择了 12_64 模型（Tfmr-scratch）与 full_bs58 模型（Tfmr-finetune）进行对比，同时三种 decoding 方式选用 random_1.0_0.0_0、random_0.85_0.0_0、random_0.7_0.0_0、top-p_1.0_0.9_0、top-p_0.7_0.9_0、top-p_1.0_0.7_0、top-k_0.7_0.0_40、top-k_1.0_0.0_40、top-k_1.0_0.0_20。




```tex
#-------------  full_bs58_random_1.0_0.0_0  -------------
A close up of a red bus driver with copter on a snowy night .
Two buses are traveling across a dirt trail .
A woman wearing a leather jacket sitting on a bench with her family in the background .
A giraffe walking past a fence and a fence back in its habitat in an outdoors setting .
A couple of white and blue buses sitting along a street .
A bus driving down a corner near red buses .
A man sits with an airplane on the runway .
An airplane parked on the tarmac at an airport .
A bright yellow fire hydrant in front of a door .
A bus that is sitting next to another bus on the side of the road .


#-------------  full_bs58_random_0.85_0.0_0  -------------
A close up of a red bus and a cop car .
A green transit bus parked along a sidewalk in front of a tree .
A woman wearing a leather jacket sitting on a bench .
A giraffe walking past a fence and a fence back in its habitat .
A couple of giraffe walking through grassy area next to tree .
A bus driving down a street near cars and buildings .
A man sits on a park bench with a motorcycle .
A woman sitting on top of a bench by herself .
A bright yellow fire hydrant in front of a large building .
A bus that is sitting in front of a truck next to a bus stop .


#-------------  full_bs58_random_0.7_0.0_0  -------------
A close up of a red bus and a cop car .
A green transit bus parked on the side of the road .
A woman sitting on a bench near a cup of coffee .
A giraffe walking past a fence and a fence post .
A couple of giraffe walking through a forest , a tree , and a few other animals .
A bus driving down a street near cars and buildings .
A man sits on a bench while waiting for the bus .
A woman sitting on top of a bench by herself .
A man is sitting on a bench with his dog .
A bus that is sitting in front of a truck .


#-------------  full_bs58_top-p_1.0_0.9_0  -------------
A close up of a red bus driver with a woman on a leash .
Two buses are traveling across a dirt trail .
A woman wearing a leather jacket sitting on a bench with her family in the background .
A giraffe walking past a fence and a fence on a sunny day .
A couple of white and blue buses sitting in front of a bunch of buildings .
A bus driving down a street near cars and buildings .
A man sits on a park bench with a motorcycle .
An airplane parked on the tarmac at an airport .
A bright yellow fire hydrant in front of a door .
A bus that is sitting next to another bus on the side of the road .


#-------------  full_bs58_top-p_0.7_0.9_0  -------------
A close up of a red bus and a red fire hydrant .
A green bench sitting on top of a grassy hill .
A woman sitting on a bench near a green bench .
A giraffe walking through a field next to a tree .
A couple of giraffe walking through a forest , a tree , and a few other animals .
A bus driving down a street near a traffic light .
A man sits on a bench while reading a book .
A woman sitting on top of a bench by a traffic light .
A man is sitting on a bench with his dog .
A bus that is sitting in front of a truck .


#-------------  full_bs58_top-p_1.0_0.7_0  -------------
A close up of a red double decker bus .
A green bench sitting on top of a grassy hill .
A woman sitting on a bench near a green bench .
A giraffe walking through a wooded area near a stone wall .
A couple of white and blue buses sitting in front of a store .
A bus driving down a street near tall buildings .
A man sits on a bench while reading a book .
A woman sitting on a bench in a park .
A bench sitting in front of a forest and some shrubs .
A bus that is sitting in front of a truck .


#-------------  full_bs58_top-k_0.7_0.0_40  -------------
A close up of a red bus and a red fire hydrant .
A green transit bus parked on the side of the road .
A woman sitting on a bench near a green bench .
A giraffe walking past a fence and a fence post .
A couple of giraffe walking through a forest , a tree , and a few other animals .
A bus driving down a street near cars and buildings .
A man sits on a bench while waiting for the bus .
A woman sitting on top of a bench by herself .
A man is sitting on a bench with his dog .
A bus that is sitting in front of a truck .


#-------------  full_bs58_top-k_1.0_0.0_40  -------------
A close up of a red bus driver with a woman on a leash wearing sunglasses .
Two buses are traveling across a dirt trail .
A woman wearing a leather jacket sitting on a bench with her family in the background .
A giraffe walking past a fence and a fence on a sunny day in a sunny part of the country .
A couple of white and blue buses sitting along a street .
A bus driving down a street near red buses .
A man sits with an airplane on the runway .
An airplane parked on the tarmac at an airport .
A bench near a grassy plain with trees in the background .
A bus that is sitting next to another bus on the side of the road .


#-------------  full_bs58_top-k_1.0_0.0_20  -------------
A group of men standing on the street with their bags on a bench .
Two buses are traveling across a dirt trail .
A woman wearing a hat sits on a bench outside .
A giraffe walking past a fence and a fence on a sunny day in a sunny part of the country .
A couple of white and blue buses sitting along a street .
A bus driving down a street near cars and buildings .
A man sits with an airplane on the runway .
An airplane parked on the tarmac at an airport .
A bench near a wall with some buildings in the background .
A bus that is sitting next to another bus on the side of the road .
```



1. 为什么 multi-head attention 比 single-head attention 的效果好？

   multi-head attention 结构设计能让每个注意力机制通过 Q、K、V 映射到不同的空间去学习特征，去优化每个词汇的不同特征部分，从而均衡同一种注意力机制可能产生的偏差。每个 attention head 最终只关注最终输出序列中一个子空间，彼此间独立，让词义拥有更多元的表达来源。实验表明 multi-head 能够显著提升模型效果。

2. **BPE** 分词器与空格分词器的区别？

   **空格分词器**也即 Word-based Tokenizer，在该分词器作用下，每个词都被分配了一个 ID，从 0 开始，一直到词汇表的大小。该模型使用这些 ID 来识别每个词。

   如果研究者希望利用 Word-based Tokenizer 完全覆盖一种语言，就需要为该语言中的每个词都有一个标识符，这将产生大量的标记。例如，英语中有超过 50 万个单词，所以要建立一个从每个单词到输入 ID 的映射，需要跟踪非常多 ID。此外，像 dog 这样的词与 dogs 这样的词的表示方法不同，模型最初将没有办法知道 dog 和 dogs 是相似的：它将识别这两个词是不相关的。同样的情况也适用于其他类似的词，比如 run 和 running，模型最初也不会认为它们是相似的。

   最后，模型需要一个自定义标记来表示不在词汇表中的单词。这就是所谓的未知标记，通常表示为 [UNK]。如果分词器产生了很多这样的标记，这通常是一个不好的迹象，因为它无法检索到一个词的合理表示，而模型正在沿途丢失信息。制作词汇的一大目标是使标记器尽可能少地将单词标记到未知标记中，而显然，Word-based Tokenizer 将面对大量的未知标记。

   BPE tokenizer 类似于 Subword-based tokenization。BPE 本身是一种简单的数据压缩算法，其中最常见的是将一对连续字节替换为该数据中不出现的字节。BPE 分词器能够达到将常用的单词在一定程度内保留，而生僻单词分解为有意义子段的目的。不常出现的词被分解成不同的 Token 参与训练，避免了大量的未知标记，也降低了词表的复杂度，有利于训练。此外，BPE tokenizer 还可以一定程度上实现 embedding sharing，前文的 dog，dogs，dogge 都可以 share dog 这个字词的 embedding，在神经网络层共享特征，也降低了训练复杂度。

3. Transformer 与 RNN 的对比？

   从张量建模的角度来看，Transformer 利用三维张量映射来建模一个 language translation 过程是非常优秀的，甚至可以说是现有算力下的极致模型。Transformer 避免了逐单词处理序列的问题，让他本身就具有了双向性。对于 CNN 与 RNN，两个不同的 token 相互作用时，模型的顺序性决定了这两个不同 token 的先后并不影响结果。而对 transformer 而言，两个不同的 token 在彼此视角下的 attention 是不同的。可以如此类比，transformer 当中的两个 token 如同两个人 Alice 和 Bob，Alice 眼中，自己和 Bob 的相互关系大概率和 Bob 眼中自己和 Alice 的相互关系是不同的，这承载了丰富的语义信息，而 CNN 和 RNN 则走向了“我见青山多妩媚，料青山见我当如是”的局限。此外，Transformer 当中的 token attention 相互关系相较 RNN 中沿着 recurrence 链条的相互关系是更加稳定的，应为 RNN 沿着序列遍历时更容易出现梯度消失或者爆炸，而 attention 能够对任意距离的 token 产生更为稳定的结果。

   从时间复杂度上，考虑 n 为序列长度，d 为 hidden layer 维度，则 transformer 的复杂度为 $O\left(n^2 d+n d^2\right)$，而RNN的时间复杂度是 $O\left(n d^2\right)$，看上去当 n 远超过 d 时，transformer 的复杂度会有显著劣势，然而 transformer 的每一层都能够高效并行，甚至还有 Megatron 这样的工作能够从系统层面优化 transformer layer 的并行计算，而 RNN 囿于模型必然的顺序性，只能够串行计算。总之，现有的加速框架能够弥合 transformer 复杂度上的不足，而相比时间复杂度的缺憾，transformer 的模型强度足够吸引人。

   