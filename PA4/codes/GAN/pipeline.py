def get_experiments():
    from itertools import product
    latent_dims = [16, 64, 100]
    hidden_dims = [16, 64, 100]
    backbones = ["CNN", "MLP"]
    return product(latent_dims, hidden_dims, backbones)

def pipeline(experiments):
    import subprocess
    for experiment in experiments:
        latent_dim, hidden_dim, backbone = experiment
        command = f"python main.py --do_train --latent_dim {latent_dim} --generator_hidden_dim {hidden_dim} --discriminator_hidden_dim {hidden_dim} --backbone {backbone} --using_wandb"
        print(command)
        subprocess.run(command, shell=True)

def test_seeds():
    import subprocess
    seeds = [43, 107, 213, 996]
    for seed in seeds:
        command = f"python main.py --do_train --latent_dim 16 --generator_hidden_dim 16 --discriminator_hidden_dim 16 --backbone CNN --using_wandb --seed {seed}"
        print(command)
        subprocess.run(command, shell=True)

if __name__ == "__main__":
    experiments = get_experiments()
    pipeline(experiments)
    test_seeds()