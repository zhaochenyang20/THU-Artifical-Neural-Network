from itertools import product
import os
import subprocess
from pathlib import Path

def train_models():
    skracth_experiments = [(3, 128), (12, 64)]
    for layer, batch_size in skracth_experiments:
        subprocess.run(
            f"python main.py --num_layers={layer} --batch_size={batch_size} --using_wandb",
            shell=True,
        )
        print(
            f"python main.py --num_layers={layer} --batch_size={batch_size} --using_wandb",
        )
    pretrained_experiments = [("./ckpt/full.tar", 58), ("./ckpt/primary.tar", 128)]
    for pretrained_ckpt, batch_size in pretrained_experiments:
        print(
            f"python main.py --pretrain_dir={pretrained_ckpt} --batch_size={batch_size} --using_wandb"
        )
        os.system(
            f"python main.py --pretrain_dir={pretrained_ckpt} --batch_size={batch_size} --using_wandb"
        )
def train_extraction():
    experiments = [1, 2, 3]
    batch_size = 128
    for experiment in experiments:
        print(
            f"python main.py --pretrain_dir=./ckpt/full.tar --extract_layer={experiment} --batch_size={batch_size} --using_wandb"
        )
        os.system(
            f"python main.py --pretrain_dir=./ckpt/full.tar --extract_layer={experiment} --batch_size={batch_size} --using_wandb"
        )

def basic_test_models(all_models, primary_bs=230, full_bs=156):
    k_s = [30, 40, 50]
    p_s = [0.7, 0.8, 0.9]
    temperatures = [0.7, 0.85, 1.0]
    choices = ["random", "top-p", "top-k"]
    experiments = [
        ("random", 1, 0, 0),
        ("random", 0.7, 0, 0),
        ("top-p", 1, 0.9, 0),
        ("top-p", 0.7, 0.9, 0),
        ("top-k", 1, 0, 40),
        ("top-k", 0.7, 0, 40),
    ]
    for decode_strategy, temperature, p, k in experiments:
        for model in all_models:
            if ("3" in model) or ("primary" in model) or ("extraction" in model):
                batch_size = primary_bs
            elif ("12" in model) or ("full" in model):
                batch_size = full_bs
            os.system(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )
            print(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )


def test_BLEU(all_models, primary_bs=230, full_bs=156):
    k_s = [30, 40, 50]
    p_s = [0.7, 0.8, 0.9]
    temperatures = [0.7, 0.85, 1.0]
    choices = ["random", "top-p", "top-k"]
    experiments = [
        # ("random", 1, 0, 0), # 这个已经做过了
        ("random", 0.85, 0, 0), # okay
        # ("random", 0.7, 0, 0), # 这个已经做过了
        # ("top-p", 1, 0.9, 0), # 这个已经做过了
        ("top-p", 1, 0.8, 0),
        ("top-p", 1, 0.7, 0),
        # ("top-k", 1, 0, 40), # 这个已经做过了
        ("top-k", 1, 0, 30), # okay
        ("top-k", 1, 0, 20), # okay
    ]
    for decode_strategy, temperature, p, k in experiments:
        for model in all_models:
            if ("3" in model) or ("primary" in model) or ("extraction" in model):
                batch_size = primary_bs
            elif ("12" in model) or ("full" in model):
                batch_size = full_bs
            os.system(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )
            print(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )

def test_ppl(primary_bs=230, full_bs=156):
    temperatures = [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6]
    models = ["./train_ckpt/3_128.tar", "./train_ckpt/12_64.tar"]
    for temperature in temperatures:
        for model in models:
            if "3" in model:
                batch_size = primary_bs
            elif "12" in model:
                batch_size = full_bs
            os.system(
                f"python main.py --test {model} --decode_strategy random --temperature={temperature} --top_p=0 --top_k=0 --batch_size={batch_size} --using_wandb"
            )
            print(
                f"python main.py --test {model} --decode_strategy random --temperature={temperature} --top_p=0 --top_k=0 --batch_size={batch_size} --using_wandb"
            )


def get_all_model_path(get_extract=False):
    model_path = Path("./train_ckpt")
    all_models = []
    for _, __, models in os.walk(model_path):
        if get_extract:
            for model in models:
                if model.endswith(".tar") and "extraction" in model:
                    model_dir = model_path / model
                    all_models.append(str(model_dir))
        else:
            for model in models:
                if model.endswith(".tar"):
                    model_dir = model_path / model
                    all_models.append(str(model_dir))
    return all_models


if __name__ == "__main__":
    # train_models()
    train_extraction()
    all_models = get_all_model_path(get_extract=True)
    basic_test_models(all_models, primary_bs=230, full_bs=156)
    test_BLEU(all_models, primary_bs=230, full_bs=156)
    # test_ppl(230, 156)
