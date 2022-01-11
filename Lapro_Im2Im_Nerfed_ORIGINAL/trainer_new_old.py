#THIS ONE HAS BEEN MODIFIED BY THE AUTHORS!
#Well...seeing that I got rid of MSSIM...is this basically just MUNIT now?
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen, StylelessGen, ResBlockSegmentation
from utils import weights_init, get_model_list, get_scheduler
from utils import __write_images as writeImage
from torch.autograd import Variable
from pytorch_msssim import msssim, ssim
import torch
import torch.nn as nn
import os
import torchvision
import random

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']

        # Initiate the networks
        self.gen_a = StylelessGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = StylelessGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        # self.gen_b = AdaINGen  (hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b

        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis']) # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis']) # discriminator for domain b

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        self.style_dim   = hyperparameters['gen']['style_dim'  ]
        self.num_classes = hyperparameters['gen']['num_classes']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        self.label_criterion = nn.CrossEntropyLoss()

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        #s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)

        c_a = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)

        x_ba = self.gen_a.decode(c_b)
        x_ab = self.gen_b.decode(c_a, s_b)

        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters, useLabelLoss=False):

        self.gen_opt.zero_grad()

        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()

        # encode
        c_a            = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)

        # Half the time, use a real style instead of a randomly drawn one:
        if random.random() > 0.5:
            s_b = s_b_prime.detach()
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b)
        x_ab = self.gen_b.decode(c_a, s_b)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)

        # Structural similarity:
        x_a_brightness  = torch.mean(x_a , dim=1, keepdim=True)
        x_b_brightness  = torch.mean(x_b , dim=1, keepdim=True)
        x_ab_brightness = torch.mean(x_ab, dim=1, keepdim=True)
        x_ba_brightness = torch.mean(x_ba, dim=1, keepdim=True)

        loss_msssim_ab = -msssim(x_a_brightness, x_ab_brightness, normalize=True)
        loss_msssim_ba = -msssim(x_b_brightness, x_ba_brightness, normalize=True)

        # encode again
        c_b_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon)            if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        loss_gen_recon_x_a    = self.recon_criterion(x_a_recon, x_a)
        loss_gen_recon_x_b    = self.recon_criterion(x_b_recon, x_b)
        loss_gen_recon_s_b    = self.recon_criterion(s_b_recon, s_b)
        loss_gen_recon_c_a    = self.recon_criterion(c_a_recon, c_a)
        loss_gen_recon_c_b    = self.recon_criterion(c_b_recon, c_b)
        loss_gen_cycrecon_x_a = self.recon_criterion(x_aba    , x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        loss_gen_cycrecon_x_b = self.recon_criterion(x_bab    , x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        # GAN loss
        loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        loss_gen_total = hyperparameters['gan_w'        ] * loss_gen_adv_a        + \
                         hyperparameters['gan_w'        ] * loss_gen_adv_b        + \
                         hyperparameters['recon_x_w'    ] * loss_gen_recon_x_a    + \
                         hyperparameters['recon_c_w'    ] * loss_gen_recon_c_a    + \
                         hyperparameters['recon_x_w'    ] * loss_gen_recon_x_b    + \
                         hyperparameters['recon_s_w'    ] * loss_gen_recon_s_b    + \
                         hyperparameters['recon_c_w'    ] * loss_gen_recon_c_b    + \
                         hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_a + \
                         hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_b + \
                         hyperparameters['ms_ssim_a_w'  ] * loss_msssim_ab        + \
                         hyperparameters['ms_ssim_b_w'  ] * loss_msssim_ba

        loss_gen_total.backward()

        self.gen_opt.step()

        self.loss_gen_adv_a        = loss_gen_adv_a       .item()
        self.loss_gen_adv_b        = loss_gen_adv_a       .item()
        self.loss_gen_recon_x_a    = loss_gen_recon_x_a   .item()
        self.loss_gen_recon_c_a    = loss_gen_recon_c_a   .item()
        self.loss_gen_recon_x_b    = loss_gen_recon_x_b   .item()
        self.loss_gen_recon_s_b    = loss_gen_recon_s_b   .item()
        self.loss_gen_recon_c_b    = loss_gen_recon_c_b   .item()
        self.loss_gen_cycrecon_x_a = loss_gen_cycrecon_x_a.item()
        self.loss_gen_cycrecon_x_b = loss_gen_cycrecon_x_b.item()
        self.loss_msssim_ab        = loss_msssim_ab       .item()
        self.loss_msssim_ba        = loss_msssim_ba       .item()

        self.loss_gen_total = loss_gen_total.item()

    def sample(self, x_a, x_b):

        self.eval()
        s_b = self.s_b
        x_a_recon, x_b_recon, x_ba, x_bab, x_ab, x_aba, x_ab_rand = [], [], [], [], [], [], []
        for i in range(x_a.size(0)):
            # get individual images from list:
            x_a_ = x_a[i].unsqueeze(0)
            x_b_ = x_b[i].unsqueeze(0)
            s_b_ = s_b[i].unsqueeze(0)

            # a to b:
            c_a        = self.gen_a.encode(x_a_)
            x_a_recon_ = self.gen_a.decode(c_a )     # Reconstruct in same domain

            c_b, s_b_extract = self.gen_b.encode(x_b_)

            x_ab_          = self.gen_b.decode(c_a, s_b_)     # translate
            c_ab, s_b_fake = self.gen_b.encode(x_ab_) # re-encode
            x_aba_         = self.gen_a.decode(c_ab ) # translate back

            x_a_recon.append(x_a_recon_)
            x_ab     .append(x_ab_)
            x_aba    .append(x_aba_    )

            # Encode another x_ab2 with a style drawn from b:
            x_ab_rand_ = self.gen_b.decode(c_a, s_b_extract)     # translate
            x_ab_rand.append( x_ab_rand_ )

            # b to a:
            x_ba_ = self.gen_a.decode(c_b)      # translate
            c_ba  = self.gen_a.encode(x_ba_)   # re-encode

            x_b_recon_ = self.gen_b.decode(c_b, s_b_extract)      # Reconstruct in same domain
            x_bab_     = self.gen_b.decode(c_ba, s_b_extract)    # translate back

            x_b_recon.append(x_b_recon_)
            x_ba     .append(x_ba_)
            x_bab    .append(x_bab_)

        x_a       =(x_a+1)/2
        x_b       =(x_b+1)/2
        x_ba      =(torch.cat(x_ba     )+1)/2
        x_ab      =(torch.cat(x_ab     )+1)/2
        x_bab     =(torch.cat(x_bab    )+1)/2
        x_aba     =(torch.cat(x_aba    )+1)/2
        x_a_recon =(torch.cat(x_a_recon)+1)/2
        x_b_recon =(torch.cat(x_b_recon)+1)/2
        x_ab_rand =(torch.cat(x_ab_rand)+1)/2

        self.train()

        return x_a, x_a_recon, x_ab, x_ab_rand, x_aba, x_b, x_b_recon, x_ba, x_bab

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        #s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)

        if random.random() > 0.5:   # With a chance of 1/2, use the style from real image
            s_b = s_b_prime.detach()

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b)
        x_ab = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])

        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt'                  )
        torch.save({'a'  : self.gen_a  .state_dict(), 'b'  : self.gen_b  .state_dict()}, gen_name)
        torch.save({'a'  : self.dis_a  .state_dict(), 'b'  : self.dis_b  .state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
