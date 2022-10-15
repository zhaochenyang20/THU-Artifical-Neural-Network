from itertools import product
import subprocess

def get_all_combinations():
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    drop_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0,7, 0.8]
    return product(batch_sizes, learning_rates, drop_rates)

def pipeline():
    for batch_size, learning_rate, drop_rate in get_all_combinations():
        subprocess.run(f"python3 main.py --batch_size {batch_size} --learning_rate {learning_rate} --drop_rate {drop_rate}", shell=True)

if __name__ == "__main__":
    pipeline()