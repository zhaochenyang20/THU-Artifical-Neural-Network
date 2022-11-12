import torch
import torch.nn as nn
import os
from torchvision.utils import make_grid
from torchvision.utils import save_image
from tqdm import tqdm


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class Trainer(object):
    def __init__(self, device, netG, netD, optimG, optimD, dataset, ckpt_dir):
        self._device = device
        self._netG = netG
        self._netD = netD
        self._optimG = optimG
        self._optimD = optimD
        self._dataset = dataset
        self._ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        self._netG.restore(ckpt_dir)
        self._netD.restore(ckpt_dir)

    def train_step(self, real_imgs, fake_imgs, BCE_criterion):
        """DO NOT FORGET TO ZERO_GRAD netD and netG
        *   Returns:
            *   loss of netD (scalar)
            *   loss of netG (scalar)
            *   average D(real_imgs) before updating netD
            *   average D(fake_imgs) before updating netD
            *   average D(fake_imgs) after updating netD
        """
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # clear gradients
        self._netD.zero_grad()

        # compute the gradients of binary_cross_entropy(netD(real_imgs), 1) w.r.t. netD
        # record average D(real_imgs)
        # TODO START
        D_x = self._netD(real_imgs)
        loss_D_real = BCE_criterion(D_x, torch.ones(D_x.shape, device=D_x.device))
        D_x = D_x.mean()
        loss_D_real.backward()
        # TODO END

        # ** accumulate ** the gradients of binary_cross_entropy(netD(fake_imgs), 0) w.r.t. netD
        # record average D(fake_imgs)
        # TODO START
        D_G_z1 = self._netD(fake_imgs)
        loss_D_fake = BCE_criterion(
            D_G_z1, torch.zeros(D_G_z1.shape, device=D_G_z1.device)
        )
        D_G_z1 = D_G_z1.mean()
        #! 总的来说进行一次 backward 之后，各个节点的值会清除，这样进行第二次 backward 会报错，如果加上 retain_graph == True 后,可以再来一次 backward
        #! accumulate means that we do not zero the gradients between two backward passes.
        loss_D_fake.backward(retain_graph=True)
        # TODO END

        # update netD
        self._optimD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # clear gradients
        self._netG.zero_grad()

        # compute the gradients of binary_cross_entropy(netD(fake_imgs), 1) w.r.t. netG
        # record average D(fake_imgs)
        # TODO START
        D_G_z2 = self._netD(fake_imgs)
        loss_G = BCE_criterion(D_G_z2, torch.ones(D_G_z2.shape, device=D_G_z2.device))
        #! However, the objective suffers from the gradient vanishing problem.
        D_G_z2 = D_G_z2.mean()
        loss_G.backward()
        # TODO END

        # update netG
        self._optimG.step()

        # return what are specified in the docstring
        return loss_D_real + loss_D_fake, loss_G, D_x, D_G_z1, D_G_z2

    def train(self, num_training_updates, logging_steps, saving_steps, using_wandb):
        import wandb

        fixed_noise = torch.randn(32, self._netG.latent_dim, 1, 1, device=self._device)
        criterion = nn.BCELoss()
        iterator = iter(cycle(self._dataset.training_loader))
        for i in tqdm(range(num_training_updates), desc="Training"):
            inp, _ = next(iterator)
            self._netD.train()
            self._netG.train()
            real_imgs = inp.to(self._device)
            fake_imgs = self._netG(
                torch.randn(
                    real_imgs.size(0), self._netG.latent_dim, 1, 1, device=self._device
                )
            )
            errD, errG, D_x, D_G_z1, D_G_z2 = self.train_step(
                real_imgs, fake_imgs, criterion
            )
            if (i + 1) % logging_steps == 0:
                if using_wandb:
                    wandb.log(
                        {
                            "discriminator_loss": errD,
                            "generator_loss": errG,
                            "D(x)": D_x,
                            "D(G(z1))": D_G_z1,
                            "D(G(z2))": D_G_z2,
                        },
                        step=i,
                    )
            if (i + 1) % saving_steps == 0:
                dirname = self._netD.save(self._ckpt_dir, i)
                dirname = self._netG.save(self._ckpt_dir, i)
                self._netG.eval()
                imgs = make_grid(self._netG(fixed_noise)) * 0.5 + 0.5
                save_image(imgs, os.path.join(dirname, "samples.png"))
