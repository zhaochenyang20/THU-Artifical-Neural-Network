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

<iframe src="https://wandb.ai/eren-zhao/Transformer-Gen/reports/Text-Generation-ANN-PA3-Report--VmlldzoyODg5OTU3" style="border:none;height:1024px;width:100%">
