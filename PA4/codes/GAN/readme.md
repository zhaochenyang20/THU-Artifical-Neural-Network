# 框架修改

我将 TensorBoard 可视化框架完全迁移到了 Wandb 框架，以此修改了实验框架。相关的效果可以参见[第一次实验结果](https://wandb.ai/eren-zhao/GAN/overview?workspace=GAN)与[第二次实验结果](https://wandb.ai/eren-zhao/finale-GAN?workspace=user-eren-zhao)。

# 运行方式

在 `./pipeline.py` 下给出了运行整体框架的方法。

1. `get_experiments`  方法用于进行实验配置；
2. `pipeline` 方法用于运行所有的实验；
3. `test_seed` 方法用于测试随机种子对于 GAN 效果的影响。

进行全部测试，直接运行如下指令即可：

```shell
python pipeline.py
```

