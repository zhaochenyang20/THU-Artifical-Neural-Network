from itertools import product
import os
import subprocess

from django.forms import modelformset_factory

def train_models():
    layers = [3, 12]
    for layer in layers:
        subprocess.run(f"python3 main.py --num_layers={layer} --using_wandb", shell=True)
    pretrained_ckpts = ["./ckpt/full.tar", "./ckpt/primary.tar"]
    for pretrained_ckpt in pretrained_ckpts:
        subprocess.run(f"python3 main.py --pretrained_dir={pretrained_ckpt} --using_wandb", shell=True)

def test_models():
    k_s = [30, 40, 50]
    p_s = [0.7, 0.8, 0.9]
    temperatures = [0.7, 0.85, 1.0]
    choices=["random", "top-p", "top-k"]
    experiments = [("random", 1, 0, 0), ("random", 0.7, 0, 0), ("top-p", 1, 0.9, 0),\
        ("top-p", 0.7, 0.9, 0), ("top-k", 1, 0, 40), ("top-k", 0.7, 0, 40)]
    from pathlib import Path
    model_path = Path("./train_ckpt")
    models = []
    for _, __, models in os.walk(model_path):
        for model in models:
            if model.endswith(".tar"):
                model_dir = model_path / model
                models.append(str(model_dir))
    for experiment in experiments:
        decode_strategy, temperature, p, k = experiment
        for model in models:
            subprocess.run(f"python3 main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k}", shell=True)

def pipeline():
    train_models()
    test_models()

if __name__ == "__main__":
    pipeline()