# Requirements

1. Implement **MLP** and **CNN** to finish the task of **cifar-10** classification under the framework of **Pytorch**.

2. Implement **Dropout**.

     <details>
     <summary><b>Dropout Implement</b></summary>
     In this code, we implement dropout in an alternative way. During the training process, we scale the remaining network nodes' output by 1/(1-p). At testing time, we do nothing in the dropout layer. It's easy to find that this method has similar results to original dropout.
     </details>

3. Implement **Batch normalization**.

4. Dataset: MLP, to load data, use `load_cifar_2d()` in `load_data.py`. CNN, to load data, use `load_cifar_4d()` in `load_data.py`.

5. MLP: Implement “input -- Linear – BN – ReLU – Dropout – Linear – loss” network in forward() in `model.py`.

6. CNN: Implement “input – Conv – BN – ReLU – Dropout – MaxPool – Conv – BN – ReLU – Dropout – MaxPool – Linear – loss” network in **forward()** in `model.py`.

7. On your final submission, you need to submit MLP and CNN codes with both **BN and dropout**. 

# Report

1. Explain how `self.training` work. Why should training and testing be different?
2. Construct the MLP and CNN with batch normalization and dropout. **Write down** the hyper-parameters that you use to obtain the best performance. **Plot** the loss value and accuracy (for both training and validation) against to every iteration (or every epoch/certain steps) during training. In summary, there are at least **2** experiments, at least **4** plots for MLP and at least **4** plots for CNN.
3. Explain why training loss and validation loss are different. How does the difference help you tuning hyper-parameters? 
4. Report the final accuracy for testing. Compare the performance of MLP and CNN and briefly explain the reason of the different performance.
5. Construct MLP and CNN without batch normalization, and discuss the effects of batch normalization (at least **2** experiments).
6. Construct MLP and CNN without dropout,  and discuss the effects of dropout (at least **2** experiments).

# Score

* Implementation: **10** points

* Report: **5** points

* Implement Dropout1d in CNN and compare the performance between Dropout1d and Dropout2d in CNN. Explain why we prefer Dropout2d in CNN.  

* Tune the hyper-parameters dropout rate, batch size. Analyze how the hyper-parameters influence the performance of dropout and batch normalization, respectively.

# Note

**NOTE:** Basic implementation should be the default setting for running because TAs will run codes to score the basic implementation. Explain in `README` how to switch to the extra implementation of bonus.

**NOTE:** The current hyper-parameter settings may not be optimal for good classification performance. Adjusting them may be helpful to get a higher accuracy.

**NOTE**: Keep at least one digit after the decimal point when you report the loss and accuracy (e.g., 56.2%).

# Submission

* **Report**: well formatted and readable summary including your results, discussions and ideas. Source codes should not be included in report writing. Only some essential lines of codes are permitted for explaining complicated thoughts. The format of a good report can be referred to a top-conference paper. **(Both Chinese and English are permitted)**
* **Codes**: organized source code files with README for **extra modifications** (other than `TODO` ) or specific usage. Ensure that others can successfully reproduce your results following your instructions. **DO NOT include model weights/raw data/compiled objects/unrelated stuff over 50MB (due to the limit of XueTang**)
* **Code Checking Result**: You should only submit the generated `summary.txt`. **DO NOT** upload any codes under `code_analysis`. However, TAs will regenerate the code checking result to ensure the correctness of the file.
