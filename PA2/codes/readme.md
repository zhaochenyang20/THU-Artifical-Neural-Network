# Readme

在实验主框架下，为了便于进行 ablation study 并 tune hyperparameter，我增加了如下的文件：

## Tuning

1. `./mlp/pipeline.py`：对 MLP 模型进行调参实验；
2. `./cnn/pipeline.py`：对 CNN 模型进行调参实验；
3. `./pipeline.py`：批量运行上方的调参实验。

## Ablation Study

1. `./mlp/ablation.py`：选取最佳 MLP 模型，测试删去 Dropout 或者 BatchNorm 的效果；
2. `./cnn/ablation.py`：选取最佳 CNN 模型，测试删去 Dropout 或者 BatchNorm 的效果；
3. `./ablation.py`：批量运行 Ablation Study。

## Dropout1d vs Dropout2d

`./cnn/dropout_test.py` 批量测试了 Dropout1d 与 Dropout2d 的效果。

