# Readme

## 接口说明

在基础功能实现的基础上，额外在 `run_mlp.py` 下添加了解析命令行参数的 `parser_data` 方法，以便于批量测试。

具体参数说明见下：

1. `-lr` 学习率
2. `-w` 消减率
3. `-b` 批量大小
4. `-n` 隐藏层个数，仅支持 0,1,2 层
5. `-a` 激活函数类别
6. `-l` 损失函数类别

使用案例可以参考 `pipeline.py` 下的 `basic_experiment` 与 `tune` 接口。

此外，为了方便做出训练图像，额外引入了 `wandb` 库。在用于提交的代码中已经将相应的操作加以注释；具体可见 `run_mlp.py` 第 105 行，`solve_net.py` 45 行与 66 行。

## 批量测试方法

```shell
python3 pipeline.py
```