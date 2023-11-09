# Readme

在实验主框架下，为了便于进行批量测试，我在 `./pipeline.py` 下设计了如下接口：

1. `train_models` 用于训练基础实验中的四个模型，两个 from scrath，另外两个 finetune
2. `train_extraction` 用于训练 `Extraciton Study` 部分的三个模型，也即从 12 层 pretrain model 当中取出 first，last，skip layers
3. `basic_test_models` 用于进行实验要求的基本测试
4. `test_BLEU`  结合 `basic_test_models` 用于测试三种 decoding strategy 对于 BLEU 和 perplexity 的影响
5. `test_ppl` 用于测试 temperature 对于 perplexity 的影响
6. `get_all_model_path` 用于得到所有有待测试的模型
7. `train_headers` 用于测试不同的 num_headers 对于 transformer 效果的影响

# 报告

**本次报告我采用了 wandb report 以方便可视化结果，由于转写 PDF 的效果不佳，烦请参阅[此链接](https://wandb.ai/eren-zhao/Transformer-Gen/reports/Text-Generation-ANN-PA3-Report--VmlldzoyODg5OTU3?accessToken=rpr9mc8yjah8t7lzrv8g149m4qur2jlxboe8sd0b7r6bc0o5lurq3bualchd6tkc)。**

# 后记

2023 年 11 月 9 日

在某次答疑过程中，我发现去年我自己的 ANN PA3 的代码居然无法复现，会在计算 forward bleu 和 backward bleu 的时候出现：

```python
File "/home/cyzhao/miniconda3/envs/prompt/lib/python3.11/site-packages/nltk/translate/bleu_score.py", line 200, in corpus_bleu
    weights[0][0]
    ~~~~~~~~~~^^^
IndexError: invalid index to scalar variable.
"""
```

其实从结果来看，这个错误还是很简单，但是我仔细捋了捋，debug 的过程比较有意思。首先，这段代码来自于多线程计算 `fw_bw_bleu` 的部分：

```python
def evaluate(gen_ids, truth_ids, cpu_count=20):
    from multiprocessing import Pool

    assert len(gen_ids) == len(truth_ids)
    sample_hyps_num = len(gen_ids)
    res = {}
    for ngrams in [4]:
        print("computing BLEU-%d"%ngrams)
        bleu_irl_fw, bleu_irl_bw = [], []
        weights = np.ones(ngrams) / ngrams

        tasks = ((truth_ids, gen_ids[i], weights) for i in range(sample_hyps_num))
        pool = Pool(cpu_count)
        values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
        values = tqdm.tqdm(values, total=sample_hyps_num)
        for ans in values:
            bleu_irl_fw.append(ans)
        pool.close()
        pool.join()

        tasks = ((gen_ids, truth_ids[i], weights) for i in range(sample_hyps_num))
        pool = Pool(cpu_count)
        values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
        values = tqdm.tqdm(values, total=sample_hyps_num)
        for ans in values:
            bleu_irl_bw.append(ans)
        pool.close()
        pool.join()

        fw_bleu = (1.0 * sum(bleu_irl_fw) / len(bleu_irl_fw))
        bw_bleu = (1.0 * sum(bleu_irl_bw) / len(bleu_irl_bw))
        if fw_bleu + bw_bleu > 0:
            fw_bw_bleu = 2.0 * bw_bleu * fw_bleu / (fw_bleu + bw_bleu)
        else:
            fw_bw_bleu = 0

        res.update({"fw-bleu-%d"%ngrams : fw_bleu, \\
            "bw-bleu-%d"%ngrams : bw_bleu, \\
            "fw-bw-bleu-%d"%ngrams : fw_bw_bleu \\
        })
    return res
```

这段代码是 ANN PA3 框架里给出的，乍一看都没有问题，语义约定也很对。我先是打断点查看了 `gen_ids` 和 `truth_ids`，是等长的 `[[ints]]`，然后这个代码看着就很正确，尝试对着异步去调试，也没有感觉到明显问题。最后往下看源代码，发现调用链如下：

`evaluate` 多线程调用 `_sentence_bleu`，`_sentence_bleu` 也是作业框架里写的，再往下调用了 `nltk` 的 `sentence_bleu`，而后 `nltk` 的 `sentence_bleu` 调用了 `corpus_bleu`，看到 `corpus_bleu` 的 199 行和 201 行：

```python
try:
        weights[0][0]
    except TypeError:
        weights = [weights]
    max_weight_length = max(len(weight) for weight in weights)
```

突然想起来我经常看到的报错就来自这个 `weights[0][0]`，但我其实还没有理解为什么错了，不过嗅觉告诉我，尝试将 `weights` 设置为缺省值看看，于是我把助教框架中的 `weights = np.ones(ngrams) / ngrams` 改成 `weights = (0.25, 0.25, 0.25, 0.25)`试一试，然后 bug 解决。

解决了 bug，但是为什么呢？`array([0.25, 0.25, 0.25, 0.25])` 和 `(0.25, 0.25, 0.25, 0.25)` 没有本质区别。于是我试着调试了下：

```python
In [6]: import numpy as np

In [7]: ngrams = 4

In [8]:         weights = np.ones(ngrams) / ngrams

In [9]:     try:
   ...:         weights[0][0]
   ...:     except TypeError:
   ...:         weights = [weights]
   ...: 
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
Cell In[9], line 2
      1 try:
----> 2     weights[0][0]
      3 except TypeError:
      4     weights = [weights]

IndexError: invalid index to scalar variable.

In [10]: weights
Out[10]: array([0.25, 0.25, 0.25, 0.25])

In [11]: weights[0][0]
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
Cell In[11], line 1
----> 1 weights[0][0]

IndexError: invalid index to scalar variable.
```

看到这个调试信息，我就明白了，这是 `nltk` 源码的鲁棒性或者说可扩展性不好。这里用一个 `try weights[0][0]` 本身就很莫名其妙了，毕竟这个函数的参数约定是一个 `tuple`，然而上来就做这个莫名其妙的操作。接着，做了这步操作后，没有把所有的 `Exception` catach 完整。框架里面传入的 `weights` 是 `array([0.25, 0.25, 0.25, 0.25])`，对于 `ndarray` 的错误是 `IndexError: invalid index to scalar variable`，所以他们没有 catch 到所有的 `Exception`，导致实际上这里必须要传入一个 `tuple (0.25, 0.25, 0.25, 0.25)` 才能被这个 try except catch 到。

这是一次很有趣的调试过程，我花费了大概一个小时，先是查看我的 inference 有没有写错，而后确信没错之后，觉得很神奇。我大概率会认为助教的代码是对的，让我很没头绪。只能继续去读源码，这才发现助教的框架和源码都有一定的问题。这没什么，毕竟源码也是工程师写的，都可能会出错。~~不过助教可能没有检查过今年的框架就让学生继续用了~~ 😂 也可能是 nltk 的某次莫名其妙的更新犯了这个工程上的错误——Graham 也教导过我，用 try except 要尽力 catch 所有的错误，而且不要用 `execept Exeception` 来忽视各种可能的错误。

这次 debug 很有意思，我学习的工程开发原则在实践中被他人犯了，而我发现这点，觉得很有成就感。

