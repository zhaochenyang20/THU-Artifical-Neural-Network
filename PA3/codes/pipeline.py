from itertools import product
import os
import subprocess

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


def basic_test_models(batch_size=160):
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
    from pathlib import Path

    model_path = Path("./train_ckpt")
    all_models = []
    for _, __, models in os.walk(model_path):
        for model in models:
            if model.endswith(".tar"):
                model_dir = model_path / model
                all_models.append(str(model_dir))
    for decode_strategy, temperature, p, k in experiments:
        for model in all_models:
            os.system(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )
            print(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )


def test_top_p(batch_size=160):
    k_s = [30, 40, 50]
    p_s = [0.7, 0.8, 0.9]
    temperatures = [0.7, 0.85, 1.0]
    choices = ["random", "top-p", "top-k"]
    experiments = [
        #! ("top-p", 1, 0.9, 0), 这个已经做过了
        ("top-p", 1, 0.8, 0),
        ("top-p", 1, 0.7, 0),
    ]
    from pathlib import Path

    model_path = Path("./train_ckpt")
    all_models = []
    for _, __, models in os.walk(model_path):
        for model in models:
            if model.endswith(".tar"):
                model_dir = model_path / model
                all_models.append(str(model_dir))
    for decode_strategy, temperature, p, k in experiments:
        for model in all_models:
            os.system(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )
            print(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )


def test_top_k(batch_size=160):
    k_s = [30, 40, 50]
    p_s = [0.7, 0.8, 0.9]
    temperatures = [0.7, 0.85, 1.0]
    choices = ["random", "top-p", "top-k"]
    experiments = [
       #! ("top-k", 1, 0, 40), 这个已经做过了
        ("top-k", 1, 0, 30),
        ("top-k", 1, 0, 20),
    ]
    from pathlib import Path

    model_path = Path("./train_ckpt")
    all_models = []
    for _, __, models in os.walk(model_path):
        for model in models:
            if model.endswith(".tar"):
                model_dir = model_path / model
                all_models.append(str(model_dir))
    for decode_strategy, temperature, p, k in experiments:
        for model in all_models:
            os.system(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )
            print(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )


def test_temperature(batch_size=160):
    k_s = [30, 40, 50]
    p_s = [0.7, 0.8, 0.9]
    temperatures = [0.7, 0.85, 1.0]
    choices = ["random", "top-p", "top-k"]
    experiments = [
       #! ("random", 1, 0, 0), 这个已经做过了
        ("random", 0.85, 0, 0),
        ("random", 0.7, 0, 0),
    ]
    from pathlib import Path

    model_path = Path("./train_ckpt")
    all_models = []
    for _, __, models in os.walk(model_path):
        for model in models:
            if model.endswith(".tar"):
                model_dir = model_path / model
                all_models.append(str(model_dir))
    for decode_strategy, temperature, p, k in experiments:
        for model in all_models:
            os.system(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )
            print(
                f"python main.py --test {model} --decode_strategy={decode_strategy} --temperature={temperature} --top_p={p} --top_k={k} --batch_size={batch_size} --using_wandb"
            )


def test_BLEU():
    batch_size = 200
    test_top_k(batch_size)
    test_top_p(batch_size)
    test_temperature(batch_size)


if __name__ == "__main__":
    test_BLEU()

