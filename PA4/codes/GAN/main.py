import GAN
from trainer import Trainer
from dataset import Dataset
from pytorch_fid import fid_score
import torch
import torch.optim as optim
import os
from torchvision.utils import make_grid
from torchvision.utils import save_image

def parser_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--latent_dim', default=16, type=int)
    parser.add_argument('--generator_hidden_dim', default=16, type=int)
    parser.add_argument('--discriminator_hidden_dim', default=16, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_training_steps', default=5000, type=int)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--data_dir', default='../data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='results', type=str, help='The path of the checkpoint directory')
    parser.add_argument("--backbone", default="CNN", choices=["CNN", "MLP"])
    parser.add_argument("--using_wandb", default=False, action='store_true')
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    return args

def manual_seed(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_wandb_running_name(args):
    latent_dim = args.latent_dim
    generator_hidden_dim = args.generator_hidden_dim
    discriminator_hidden_dim = args.discriminator_hidden_dim
    #* usually discriminator_hidden_dim equals discriminator_hidden_dim
    backbone = args.backbone
    seed = args.seed
    manual_seed(seed)
    if seed != 42:
        print(f"GAN_{backbone}_latent_dim_{latent_dim}_generator_hidden_dim_{generator_hidden_dim}_discriminator_hidden_dim_{discriminator_hidden_dim}_seed_{seed}")
        wandb_run_name = f"{backbone}_{latent_dim}_{generator_hidden_dim}_{discriminator_hidden_dim}_{seed}"
    else:
        print(f"GAN_{backbone}_latent_dim_{latent_dim}_generator_hidden_dim_{generator_hidden_dim}_discriminator_hidden_dim_{discriminator_hidden_dim}")
        wandb_run_name = f"{backbone}_{latent_dim}_{generator_hidden_dim}_{discriminator_hidden_dim}"
    return wandb_run_name

if __name__ == "__main__":
    args = parser_data()
    wandb_run_name = get_wandb_running_name(args)
    using_wandb = args.using_wandb
    if using_wandb:
        import wandb
        wandb.init(project="GAN", entity="eren-zhao", name=wandb_run_name)
        wandb.config = {
            "latent_dim": args.latent_dim,
            "generator_hidden_dim": args.generator_hidden_dim,
            "discriminator_hidden_dim": args.discriminator_hidden_dim,
            "batch_size": args.batch_size,
            "num_training_steps": args.num_training_steps,
            "learning_rate": args.learning_rate,
            "beta1": args.beta1,
            "backbone": args.backbone,
            "seed": args.seed,
        }
    args.ckpt_dir = os.path.join(args.ckpt_dir, wandb_run_name)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    dataset = Dataset(args.batch_size, args.data_dir)
    netG, netD = GAN.get_model(1, args.latent_dim, args.generator_hidden_dim, args.backbone, device)
    if args.do_train:
        optimG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        optimD = optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        trainer = Trainer(device, netG, netD, optimG, optimD, dataset, args.ckpt_dir)
        trainer.train(args.num_training_steps, args.logging_steps, args.saving_steps, using_wandb)

        netG.eval()
        for pair in range(5):
            z1 = torch.randn(size = (args.latent_dim, 1, 1), device = device)
            z2 = torch.randn(size = (args.latent_dim, 1, 1), device = device)
            all_z = [z1 + i / 10 * (z2 - z1) for i in range(10)]
            all_z = torch.stack(all_z)
            save_image(make_grid(netG(all_z), nrow = 5, normalize = True, value_range = (-1, 1)), os.path.join(args.ckpt_dir, "interpolate_{}.png".format(pair)))

        save_image(make_grid(netG(torch.randn(size = (50, args.latent_dim, 1, 1), device = device)),
                                nrow = 10, normalize = True, value_range = (-1, 1)),
                    os.path.join(args.ckpt_dir, "Linear_Interpolation.png"))
    steps = [each for each in os.listdir(args.ckpt_dir) if not each.endswith(".png")]
    restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in steps)))
    netG.restore(restore_ckpt_path)

    num_samples = 3000
    real_imgs = None
    real_dl = iter(dataset.training_loader)
    while real_imgs is None or real_imgs.size(0) < num_samples:
        imgs = next(real_dl)
        if real_imgs is None:
            real_imgs = imgs[0]
        else:
            real_imgs = torch.cat((real_imgs, imgs[0]), 0)
    real_imgs = real_imgs[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5

    with torch.no_grad():
        samples = None
        while samples is None or samples.size(0) < num_samples:
            imgs = netG.forward(torch.randn(args.batch_size, netG.latent_dim, 1, 1, device=device))
            if samples is None:
                samples = imgs
            else:
                samples = torch.cat((samples, imgs), 0)
    samples = samples[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5
    samples = samples.cpu()

    fid = fid_score.calculate_fid_given_images(real_imgs, samples, args.batch_size, device)
    if using_wandb:
        wandb.log({"fid": fid})
    print("FID score: {:.3f}".format(fid), flush=True)