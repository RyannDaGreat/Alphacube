#THIS ONE HAS BEEN MODIFIED BY THE AUTHORS!
#Well...seeing that I got rid of MSSIM...is this basically just MUNIT now?
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen, StylelessGen, ResBlockSegmentation
from utils import weights_init, get_model_list, get_scheduler
from utils import __write_images as writeImage
from pytorch_msssim import msssim, ssim
from torch.autograd import Variable

import os
import torch
import random
import torch.nn as nn

import upper.source.projector          as projector
import upper.source.unprojector        as unprojector
import upper.source.scene_reader       as scene_reader
import upper.source.view_consistency   as view_consistency
import upper.source.learnable_textures as learnable_textures


label_values = [0,255]
texture_loss_weight = 20
view_consistency_version = 'std'
texture_multiplier = 5
texture_reality_loss_weight = 2 #Gotta rename this; it controls how much the texture and translations have to match

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters, trainable=True):

        super().__init__()

        if trainable:
            #If self.trainable is True, we also allocate discriminators and optimizers etc to vram
            self.trainable=True
        else:
            self.trainable=False

        self.hyperparameters=hyperparameters

        #TODO: Connect the config to change the height, width, num_channels etc of the learnable textures
        self.texture_pack = learnable_textures.LearnableTexturePackFourier(height=256,width=256,num_textures=len(label_values)) 

        a_num_channels = hyperparameters['input_dim_a']#+self.texture_pack.num_channels
        b_num_channels = hyperparameters['input_dim_b']
        self.view_consistency_loss = view_consistency.ViewConsistencyLoss(recovery_width = 128, recovery_height = 128, version = view_consistency_version)

        print("BATCH SIZE",hyperparameters['batch_size'])
        if not hyperparameters['batch_size']>1:print( "batch_size must be MORE than 1, but its %i"%hyperparameters['batch_size'])

        if self.trainable:
            lr = hyperparameters['lr']

        # Initiate the networks
        self.gen_a = StylelessGen(a_num_channels, hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = StylelessGen(b_num_channels, hyperparameters['gen'])  # auto-encoder for domain b
        # self.gen_b = AdaINGen  (hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b

        if self.trainable:
            self.dis_a = MsImageDis(a_num_channels, hyperparameters['dis']) # discriminator for domain a
            self.dis_b = MsImageDis(b_num_channels, hyperparameters['dis']) # discriminator for domain b

        # self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # self.style_dim   = hyperparameters['gen']['style_dim'  ]
        self.num_classes = hyperparameters['gen']['num_classes']

        # fix the noise used in sampling
        # display_size = int(hyperparameters['display_size'])
        # self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        if self.trainable:
            beta1 = hyperparameters['beta1']
            beta2 = hyperparameters['beta2']

            dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
            gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())

            dis_params = [p for p in dis_params if p.requires_grad]
            gen_params = [p for p in gen_params if p.requires_grad]
            tex_params = list(self.texture_pack.parameters())

            self.dis_opt = torch.optim.Adam(dis_params, lr=lr   , betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
            self.gen_opt = torch.optim.Adam(gen_params, lr=lr   , betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
            self.tex_opt = torch.optim.Adam(tex_params, lr=lr*10, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

            self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
            self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
            self.tex_scheduler = get_scheduler(self.tex_opt, hyperparameters)

        # Network weight initialization
        if self.trainable:
            self.apply(weights_init(hyperparameters['init']))
            self.dis_a.apply(weights_init('gaussian'))
            self.dis_b.apply(weights_init('gaussian'))

            self.label_criterion = nn.CrossEntropyLoss()


    def project_texture_pack(self, x_a):
        scene_uvs, scene_labels = scene_reader.extract_scene_uvs_and_scene_labels(scene_images = x_a         ,
                                                                                  label_values = label_values)

        texture_pack=self.texture_pack()

        scene_projections = projector.project_textures(scene_uvs, scene_labels, texture_pack)

        #TOO MANY CHANNELS:
        # x_a = x_a * 2 - 1 #Convert from 0,1 range to -1,1 range
        # x_a = torch.stack((x_a,scene_projections), dim=1) #Add projected textures
        # that involves creating more visualizations for all 6 channels and textures

        #SIMPLE:
        x_a=x_a*2-1

        content = x_a+0 #Might replace with better content later

        #PURE LEARNED:
        # x_a = scene_projections #let's try to minimize effort right now...let's just use 3 channels for visualization etc... todo make all 6:

        #COMPROMISE:
        # x_a[:,:2] = scene_projections[:,:2] #A compromise...I'll force the third channel (blue) to be preserved and the first two channels (R,G) are learnable. I'll add more channels later...but right now I don't have the visualizers for this.

        #RESIDUAL:
        x_a = x_a + (scene_projections-1/2)*2*texture_multiplier#let's try to minimize effort right now...let's just use 3 channels for visualization etc... todo make all 6:

        x_a = torch.cat((x_a,content),dim=1)#BCHW

        return x_a, scene_uvs, scene_labels


    def recon_criterion(self, input, target):
        # return -msssim(input,target,normalize=True)
        return torch.mean(torch.abs(input - target))


    def forward(self, x_a, x_b):
        x_a, _, _ = self.project_texture_pack(x_a)

        self.eval()
        #s_a = Variable(self.s_a)
        # s_b = Variable(self.s_b)

        c_a = self.gen_a.encode(x_a)
        c_b = self.gen_b.encode(x_b)
        # c_b, s_b_fake = self.gen_b.encode(x_b)

        x_ba = self.gen_a.decode(c_b)
        x_ab = self.gen_b.decode(c_a)
        # x_ab = self.gen_b.decode(c_a, s_b)

        self.train()
        return x_ab, x_ba


    def gen_update(self, x_a, x_b, hyperparameters, useLabelLoss=False):

        assert self.trainable, 'This MUNIT_Trainer is not trainable, and does not have optimizers or discriminators etc'

        ###########################
        ####### RYAN'S CODE #######
        ###########################

        #Because precise=True, x_a should be in the range (0,1) and x_b should be in the range (-1,1) because precise=False for that domain (see utils.py)

        x_a, scene_uvs, scene_labels = self.project_texture_pack(x_a)

        self.tex_opt.zero_grad()


        #############################
        ####### ORIGINAL CODE #######
        #############################
        
        self.gen_opt.zero_grad()

        # s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()

        # encode
        c_a = self.gen_a.encode(x_a)
        c_b = self.gen_b.encode(x_b)
        # c_b, s_b_prime = self.gen_b.encode(x_b)

        # Half the time, use a real style instead of a randomly drawn one:
        # if random.random() > 0.5:
        #     s_b = s_b_prime.detach()
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b)
        x_ab = self.gen_b.decode(c_a)
        # x_ab = self.gen_b.decode(c_a, s_b)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a)
        x_b_recon = self.gen_b.decode(c_b)
        # x_b_recon = self.gen_b.decode(c_b, s_b_prime)

        # Structural similarity:
        # x_a_brightness  = torch.mean(x_a , dim=1, keepdim=True)
        # x_b_brightness  = torch.mean(x_b , dim=1, keepdim=True)
        # x_ab_brightness = torch.mean(x_ab, dim=1, keepdim=True)
        # x_ba_brightness = torch.mean(x_ba, dim=1, keepdim=True)

        # loss_msssim_ab = -msssim(x_a_brightness, x_ab_brightness, normalize=True)
        # loss_msssim_ba = -msssim(x_b_brightness, x_ba_brightness, normalize=True)

        # encode again
        c_b_recon = self.gen_a.encode(x_ba)
        c_a_recon = self.gen_b.encode(x_ab)
        # c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon)            if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon)            if hyperparameters['recon_x_cyc_w'] > 0 else None
        # x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        loss_gen_recon_x_a    = self.recon_criterion(x_a_recon, x_a)
        loss_gen_recon_x_b    = self.recon_criterion(x_b_recon, x_b)
        # loss_gen_recon_s_b    = self.recon_criterion(s_b_recon, s_b)
        loss_gen_recon_c_a    = self.recon_criterion(c_a_recon, c_a)
        loss_gen_recon_c_b    = self.recon_criterion(c_b_recon, c_b)
        loss_gen_cycrecon_x_a = self.recon_criterion(x_aba    , x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        loss_gen_cycrecon_x_b = self.recon_criterion(x_bab    , x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        # GAN loss
        loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        
        #View Consistency Loss
        loss_view_consistency = self.view_consistency_loss(x_ab, scene_uvs, scene_labels)
        # loss_view_consistency = 0
        # print(float(loss_view_consistency))

        # if (loss_view_consistency.isnan() | loss_view_consistency.isinf()).any(): print("view consistency has nan or inf")

        #Total loss
        loss_gen_total  = hyperparameters['gan_w'        ] * loss_gen_adv_a       
        loss_gen_total += hyperparameters['gan_w'        ] * loss_gen_adv_b       
        loss_gen_total += hyperparameters['recon_x_w'    ] * loss_gen_recon_x_a   
        loss_gen_total += hyperparameters['recon_c_w'    ] * loss_gen_recon_c_a   
        loss_gen_total += hyperparameters['recon_x_w'    ] * loss_gen_recon_x_b   
        loss_gen_total += hyperparameters['recon_c_w'    ] * loss_gen_recon_c_b   
        loss_gen_total += hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_a
        loss_gen_total += hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_b
        loss_gen_total += texture_loss_weight              * loss_view_consistency
        # loss_gen_total += hyperparameters['ms_ssim_a_w'  ] * loss_msssim_ab
        # loss_gen_total += hyperparameters['ms_ssim_b_w'  ] * loss_msssim_ba
        # loss_gen_total += hyperparameters['recon_s_w'    ] * loss_gen_recon_s_b

        
        texture_reality_loss =(      (x_ab-x_a[:,:3])**2).mean()    *texture_reality_loss_weight
        texture_reality_loss+=-msssim(x_ab,x_a[:,:3],normalize=True)*texture_reality_loss_weight #normalize=True because without it we get tons of NaN's

        loss_gen_total = loss_gen_total + texture_reality_loss

        loss_gen_total.backward()

        self.tex_opt.step()
        self.gen_opt.step()


        #Unimportant code:
        self.loss_gen_adv_a        = loss_gen_adv_a       .item()
        self.loss_gen_adv_b        = loss_gen_adv_a       .item()
        self.loss_gen_recon_x_a    = loss_gen_recon_x_a   .item()
        self.loss_gen_recon_c_a    = loss_gen_recon_c_a   .item()
        self.loss_gen_recon_x_b    = loss_gen_recon_x_b   .item()
        # self.loss_gen_recon_s_b    = loss_gen_recon_s_b   .item()
        self.loss_gen_recon_c_b    = loss_gen_recon_c_b   .item()
        self.loss_gen_cycrecon_x_a = loss_gen_cycrecon_x_a.item()
        self.loss_gen_cycrecon_x_b = loss_gen_cycrecon_x_b.item()
        # self.loss_msssim_ab        = loss_msssim_ab       .item()
        # self.loss_msssim_ba        = loss_msssim_ba       .item()
        self.loss_gen_total = loss_gen_total.item()


    def sample(self, x_a, x_b, with_grad=False):

        if not with_grad:
            with torch.no_grad():
                return self.sample(x_a, x_b, with_grad=True)

        x_a_original = x_a #This is the UVL map

        x_a, _, _ = self.project_texture_pack(x_a)

        self.eval()
        # s_b = self.s_b
        x_a_recon, x_b_recon, x_ba, x_bab, x_ab, x_aba, x_ab_rand = [], [], [], [], [], [], []
        for i in range(x_a.size(0)):
            # get individual images from list:
            x_a_ = x_a[i].unsqueeze(0)
            x_b_ = x_b[i].unsqueeze(0)
            # s_b_ = s_b[i].unsqueeze(0)

            # a to b:
            c_a        = self.gen_a.encode(x_a_)
            x_a_recon_ = self.gen_a.decode(c_a )     # Reconstruct in same domain

            # c_b, s_b_extract = self.gen_b.encode(x_b_)
            c_b= self.gen_b.encode(x_b_)

            # x_ab_          = self.gen_b.decode(c_a, s_b_)     # translate
            x_ab_          = self.gen_b.decode(c_a   ) # translate
            c_ab           = self.gen_b.encode(x_ab_ ) # re-encode
            # c_ab, s_b_fake = self.gen_b.encode(x_ab_) # re-encode
            x_aba_         = self.gen_a.decode(c_ab  ) # translate back

            x_a_recon.append(x_a_recon_ )
            x_ab     .append(x_ab_      )
            x_aba    .append(x_aba_     )

            # Encode another x_ab2 with a style drawn from b:
            x_ab_rand_ = self.gen_b.decode(c_a)     # translate
            # x_ab_rand_ = self.gen_b.decode(c_a, s_b_extract)     # translate
            x_ab_rand.append( x_ab_rand_ )

            # b to a:
            x_ba_ = self.gen_a.decode(c_b   ) # translate
            c_ba  = self.gen_a.encode(x_ba_ ) # re-encode

            # x_b_recon_ = self.gen_b.decode(c_b, s_b_extract)      # Reconstruct in same domain
            # x_bab_     = self.gen_b.decode(c_ba, s_b_extract)    # translate back
            x_b_recon_ = self.gen_b.decode(c_b  ) # Reconstruct in same domain
            x_bab_     = self.gen_b.decode(c_ba ) # translate back

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

        return x_a_original, x_a[:,:3], x_a_recon[:,:3], x_a_recon[:,3:], x_ab, x_aba[:,:3], x_aba[:,3:], \
               x_b, x_b_recon, x_ba[:,:3], x_ba[:,3:], x_bab
        # return x_a_original, x_a, x_a_recon, x_ab, x_ab_rand, x_aba, x_b, x_b_recon, x_ba, x_bab #We removed all randomness, so x_ab_rand==x_ab exactly (I tested it - it's true. They're identical and therefore redundant)

    def sample_a2b(self, x_a, with_grad=False):
        #This code is very similar to self.sample(), except it has a few parts removed for the sake of efficiency.
        #If you ever want to modify the functionality of this function, make sure you modify it in self.sample() too
        #TODO: Remove this redundancy lol

        if not with_grad:
            with torch.no_grad():
                return self.sample_a2b(x_a, with_grad=True)

        x_a, _, _ = self.project_texture_pack(x_a)

        self.eval()
        x_ab = []

        for i in range(x_a.size(0)):
            # get individual images from list:
            x_a_ = x_a[i].unsqueeze(0)

            c_a   = self.gen_a.encode(x_a_)
            x_ab_ = self.gen_b.decode(c_a )

            x_ab.append(x_ab_)

        x_ab=(torch.cat(x_ab)+1)/2

        self.train()

        return x_ab


    def dis_update(self, x_a, x_b, hyperparameters):

        assert self.trainable, 'This MUNIT_Trainer is not trainable, and does not have optimizers or discriminators etc'

        x_a, _, _ = self.project_texture_pack(x_a)

        self.dis_opt.zero_grad()
        #s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a = self.gen_a.encode(x_a)
        c_b = self.gen_b.encode(x_b)

        # if random.random() > 0.5:   # With a chance of 1/2, use the style from real image
        #     s_b = s_b_prime.detach()

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b)
        x_ab = self.gen_b.decode(c_a)
        # x_ab = self.gen_b.decode(c_a, s_b)
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
        if self.tex_scheduler is not None:
            self.tex_scheduler.step()


    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])

        # Load textures
        last_model_name = get_model_list(checkpoint_dir, "tex")
        state_dict = torch.load(last_model_name)
        self.texture_pack.load_state_dict(state_dict['tex'])

        if self.trainable:
            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis")
            state_dict = torch.load(last_model_name)
            self.dis_a.load_state_dict(state_dict['a'])
            self.dis_b.load_state_dict(state_dict['b'])

            # Load optimizers
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.dis_opt.load_state_dict(state_dict['dis'])
            self.gen_opt.load_state_dict(state_dict['gen'])
            self.tex_opt.load_state_dict(state_dict['tex'])

            # Reinitilize schedulers
            self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
            self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
            self.tex_scheduler = get_scheduler(self.tex_opt, hyperparameters, iterations)
            print('Resume from iteration %d' % iterations)
            return iterations


    def save(self, snapshot_dir, iterations):

        assert self.trainable, 'This MUNIT_Trainer is not trainable, and does not have optimizers or discriminators etc'

        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        tex_name = os.path.join(snapshot_dir, 'tex_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt'                  )

        torch.save({'a'  : self.gen_a       .state_dict(), 'b'  : self.gen_b  .state_dict()                                  }, gen_name)
        torch.save({'a'  : self.dis_a       .state_dict(), 'b'  : self.dis_b  .state_dict()                                  }, dis_name)
        torch.save({'tex': self.texture_pack.state_dict()                                                                    }, tex_name)
        torch.save({'gen': self.gen_opt     .state_dict(), 'dis': self.dis_opt.state_dict(), 'tex': self.tex_opt.state_dict()}, opt_name)



