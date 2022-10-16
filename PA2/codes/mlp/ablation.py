from itertools import product
import subprocess

def get_all_combinations():
    batch_sizes = [1024]
    learning_rates = [0.001]
    drop_rates = [0.2]
    return product(batch_sizes, learning_rates, drop_rates)

def pipeline():
    for batch_size, learning_rate, drop_rate in get_all_combinations():
        subprocess.run(f"python3 main.py --batch_size {batch_size} --learning_rate {learning_rate} --drop_rate {drop_rate}", shell=True)
        subprocess.run(f"python3 main.py --batch_size {batch_size} --learning_rate {learning_rate} --drop_rate {drop_rate} --without_BatchNorm", shell=True)
        subprocess.run(f"python3 main.py --batch_size {batch_size} --learning_rate {learning_rate} --drop_rate {drop_rate} --without_dropout", shell=True)
        subprocess.run(f"python3 main.py --batch_size {batch_size} --learning_rate {learning_rate} --drop_rate {drop_rate} --without_BatchNorm --without_dropout", shell=True)

if __name__ == "__main__":
    pipeline()