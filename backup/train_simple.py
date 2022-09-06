'''
This is a simplified training code of GPEN. It achieves comparable performance as in the paper.

@Created by rosinality

@Modified by yangxy (yangtao9009@gmail.com)
'''
import argparse
import math
import random
import os
import cv2
import glob
from tqdm import tqdm

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils

import __init_paths
from training.data_loader.dataset_face import FaceDataset, RefFaceDataset
from face_model.gpen_model import FullGenerator, RefineNetwork

#from training.loss.id_loss import IDLoss
from mmedit.models.losses.gan_loss import GANLoss
from mmedit.models.components.discriminators.unet_disc import UNetDiscriminatorWithSpectralNorm
from mmedit.models.losses.gan_loss import GANLoss
from mmedit.models.losses.perceptual_loss import PerceptualLoss

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from training import lpips


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def validation(first_stage_generator, g_ema, ref_img, lpips_func, args, device):
    lq_files = sorted(glob.glob(os.path.join(args.val_dir, 'lq', '*.*')))
    hq_files = sorted(glob.glob(os.path.join(args.val_dir, 'hq', '*.*')))

    assert len(lq_files) == len(hq_files)

    dist_sum = 0
    g_ema.eval()
    for lq_f, hq_f in zip(lq_files, hq_files):
        img_lq = cv2.imread(lq_f, cv2.IMREAD_COLOR)
        img_t = torch.from_numpy(img_lq).to(device).permute(2, 0, 1).unsqueeze(0)
        img_t = (img_t/255.-0.5)/0.5
        img_t = F.interpolate(img_t, (args.size, args.size))
        img_t = torch.flip(img_t, [1])
        
        with torch.no_grad():
            #img_out, __ = model(img_t)
            Isisr, _ = first_stage_generator(img_t)
            img_out, _ = g_ema(Isisr, ref_img)
        
            img_hq = lpips.im2tensor(lpips.load_image(hq_f)).to(device)
            img_hq = F.interpolate(img_hq, (args.size, args.size))
            dist_sum += lpips_func.forward(img_out, img_hq)
    
    return dist_sum.data/len(lq_files)


def train(args, loader, first_stage_generator, second_stage_refine_network, discriminator, losses, g_optim, d_optim, g_ema, lpips_func, device):
    #loader = sample_data(loader)
    LQ_loader, Ref_loader = loader
    LQ_loader = sample_data(LQ_loader)
    Ref_loader = sample_data(Ref_loader)

    smooth_l1_loss, AdvLoss, PercepLoss = losses

    requires_grad(first_stage_generator, False)

    pbar = range(0, args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    """
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    """
    loss_dict = {}

    if args.distributed:
        g_module = second_stage_refine_network.module
        d_module = discriminator.module
    else:
        g_module = second_stage_refine_network
        d_module = discriminator
 
    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break
        
        #load data
        degraded_img, real_img = next(LQ_loader)
        degraded_img = degraded_img.to(device)
        real_img = real_img.to(device)

        ref_img = next(Ref_loader)
        ref_img = ref_img.to(device)

        #Train Discriminator 
        requires_grad(second_stage_refine_network, False)
        requires_grad(discriminator, True)

        #fake_img, _ = generator(degraded_img)
        Isisr, _ = first_stage_generator(degraded_img)
        IrefSR, _ = second_stage_refine_network(Isisr, ref_img)

        #real
        real_d_pred = discriminator(real_img)
        loss_d_real = AdvLoss(real_d_pred, target_is_real=True, is_disc=True)

        #fake
        fake_d_pred = discriminator(IrefSR.detach())
        loss_d_fake = AdvLoss(fake_d_pred, target_is_real=False, is_disc=True)

        #update
        d_loss = loss_d_real.mean() + loss_d_fake.mean()

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_d_pred.mean()
        loss_dict['fake_score'] = fake_d_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        #Train Generator
        requires_grad(second_stage_refine_network, True)
        requires_grad(discriminator, False)

        #fake_img, _ = generator(degraded_img)
        Isisr, _ = first_stage_generator(degraded_img)
        IrefSR, Itex = second_stage_refine_network(Isisr, ref_img)

        #Gan Loss
        fake_g_pred = discriminator(IrefSR)
        loss_gan = AdvLoss(fake_g_pred, target_is_real=True, is_disc=False)

        #perceptual loss
        loss_percep, _ = PercepLoss(IrefSR, real_img)

        #rec loss
        loss_rec = smooth_l1_loss(IrefSR, real_img)

        #Itex loss
        loss_Itex = smooth_l1_loss(real_img, (Isisr+Itex))

        #update
        g_loss = 1.0*loss_gan.mean() + 0.01*loss_percep.mean() + 1.0*loss_rec + 1.0*loss_Itex
        loss_dict['g'] = g_loss

        #generator.zero_grad()
        second_stage_refine_network.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f};'
                )
            )
            
            if i % args.save_freq == 0:
                with torch.no_grad():
                    g_ema.eval()
                    #sample, _ = g_ema(degraded_img)
                    Isisr, _ = first_stage_generator(degraded_img)
                    sample, Itex = g_ema(Isisr, ref_img)

                    sample = torch.cat((degraded_img, sample, real_img), 0) 
                    utils.save_image(
                        sample,
                        f'{args.sample}/{str(i).zfill(6)}_result_{args.name}.png',
                        nrow=args.batch,
                        normalize=True,
                        range=(-1, 1),
                    )
            
                    misc_img = torch.cat((Isisr, Itex, real_img, ref_img), 0) 
                    utils.save_image(
                        misc_img,
                        f'{args.sample}/{str(i).zfill(6)}_misc_{args.name}.png',
                        nrow=args.batch,
                        normalize=True,
                        range=(-1, 1),
                    )

                lpips_value = validation(first_stage_generator, g_ema, ref_img, lpips_func, args, device)
                print(f'{i}/{args.iter}: lpips: {lpips_value.cpu().numpy()[0][0][0][0]}')

            if i and i % args.save_freq == 0:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    f'{args.ckpt}/{str(i).zfill(6)}_{args.name}.pth',
                )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True)
    #parser.add_argument('--path', type=str, required=False)
    parser.add_argument('--base_dir', type=str, default='./')
    parser.add_argument('--iter', type=int, default=4000000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='ckpts')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--sample', type=str, default='sample')
    parser.add_argument('--val_dir', type=str, default='val')
    parser.add_argument('--ref_dir', type=str, default='ref')
    parser.add_argument('--name', type=str, default='model')

    args = parser.parse_args()
    """
    args.path = "./train"
    args.batch = 1
    args.sample = 'results'
    args.ckpt = 'weights'
    """
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.sample, exist_ok=True)

    device = 'cuda'

    args.size = 512

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8
    
    args.start_iter = 0

    #first stage generator
    first_stage_generator = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    )
    #########################################
    ckpts = torch.load("./weights/GPEN-BFR-512.pth", map_location="cpu")
    first_stage_generator.load_state_dict(ckpts)
    first_stage_generator = first_stage_generator.to(device)
    requires_grad(first_stage_generator, False)
    #########################################

    #second stage generator
    second_stage_refine_network = RefineNetwork().to(device)

    #Discriminator
    discriminator = UNetDiscriminatorWithSpectralNorm(in_channels=3,mid_channels=64,skip_connection=True).to(device)

    #ema generator
    g_ema = RefineNetwork().to(device)
    g_ema.eval()
    accumulate(g_ema, second_stage_refine_network, 0)
    
    g_optim = optim.Adam(
        second_stage_refine_network.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
    )
    """
    if args.pretrain is not None:
        print('load model:', args.pretrain)
        
        ckpt = torch.load(args.pretrain)

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])
            
        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
    """
    smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)
    
    AdvLoss = GANLoss(gan_type='vanilla',loss_weight=1.0,real_label_val=1.0,fake_label_val=0).to(device)
    PercepLoss =  PerceptualLoss(
        layer_weights={'2': 0.1,'7': 0.1,'16': 1.0,'25': 1.0,'34': 1.0,},
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=True
        ).to(device)
    lpips_func = lpips.LPIPS(net='alex',version='0.1').to(device)
    
    if args.distributed:
        second_stage_refine_network = nn.parallel.DistributedDataParallel(
            second_stage_refine_network,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        AdvLoss = nn.parallel.DistributedDataParallel(
            AdvLoss,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        PercepLoss = nn.parallel.DistributedDataParallel(
            PercepLoss,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    dataset = FaceDataset(args.path, args.size)
    LQ_loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    Refdataset = RefFaceDataset(args.ref_dir, args.size)
    Ref_loader = data.DataLoader(
        Refdataset,
        batch_size=args.batch,
        sampler=data_sampler(Refdataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    loader = [LQ_loader, Ref_loader]

    train(args, loader, first_stage_generator, second_stage_refine_network, discriminator, [smooth_l1_loss, AdvLoss, PercepLoss], g_optim, d_optim, g_ema, lpips_func, device)
   
