import os
import sys
import pdb
import glob
import time
import random
import pickle
import logging
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import cv2

from main import parse_args_and_config
from guided_diffusion.diffusion import Diffusion
from guided_diffusion.diffusion import compute_alpha



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  


def get_data(args):
    train_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    train_dataset = torchvision.datasets.ImageFolder(os.path.join("datasets/small_celeba_hq", "train"), transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join("datasets/small_celeba_hq", "val"), transform=val_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    return train_dataset, train_dataloader, val_dataset, val_dataloader

def save_checkpoint(state, epoch, mse, psnr, ssim, save_dir, filename='checkpoint.pth'):
    filename = os.path.join(save_dir, 'Epoch_%d_MSE_%.4f_PSNR_%.2f_SSIM_%.2f_'%(epoch, mse, psnr, ssim) + filename)
    torch.save(state, filename)


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
  
def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))    




class Trainer(object):
    def __init__(self, diffusion, criterion=None, optimizer=None, scheduler=None, args=None, logging=None, device=None):
        self.diffusion = diffusion
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.logging = logging
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def train(self, train_loader, epoch, args=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mse_losses = AverageMeter()
        psnrs = AverageMeter()

        # switch to train mode
        self.diffusion.adapter.train()

        lr = self.scheduler.get_last_lr()[0]
        self.logging.info('Epoch {:3d} lr = {:.6e}'.format(epoch, lr))

        start = time.time()
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if i % args.train_size != 0:
                continue
            if i > args.num_trainset:
                continue
            data_time.update(time.time() - start)
            start = time.time()
            # measure data loading time
            images = images.float()
            images = images.to(self.device)

            t = torch.randint(0, self.diffusion.num_timesteps, (images.shape[0],), device=self.device).long()  # random t 0-1000, size = (1)
            noise = torch.randn_like(images)
            x_t = self.diffusion.q_sample(x_start=images, t=t, noise=noise)
            if "unet" in args.A_type:
                predicted_noise = self.diffusion.model(x_t, t, self.diffusion.adapter, args)
            else:
                predicted_noise = self.diffusion.model(x_t, t)
            predicted_noise = predicted_noise[:, :3]

            at = compute_alpha(self.diffusion.betas, t.long())
            x0_t = (x_t - predicted_noise * (1 - at).sqrt()) / at.sqrt()

            if args.simplified:
                x0_t_hat = x0_t - self.diffusion.adapter(self.diffusion.Ap(self.diffusion.A(x0_t - images)), t)
            elif args.A_type == 'adapter':
                x0_t_hat = x0_t - self.diffusion.adapter(self.diffusion.A_funcs.A_pinv(
                        self.diffusion.A_funcs.A(x0_t.reshape(x0_t.size(0), -1) - images.reshape(images.size(0), -1))
                    ).reshape(*x0_t.size()), t)
            else:
                x0_t_hat = x0_t - self.diffusion.A_funcs.A_pinv(
                        self.diffusion.A_funcs.A(x0_t.reshape(x0_t.size(0), -1) - images.reshape(images.size(0), -1))
                    ).reshape(*x0_t.size())

            mse_loss = 0

            for j in range(len(x0_t_hat)):
                mse_loss +=  self.criterion(x0_t_hat[j], images[j])

            mse_loss /= len(x0_t_hat)
            loss = mse_loss

            # compute gradient and do ADAM step
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_max_norm>0:
                torch.nn.utils.clip_grad_norm_(self.diffusion.adapter.parameters(), args.clip_max_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # measure error and record loss
            losses.update(loss.item())
            mse_losses.update(mse_loss.item())
            psnr = 10*np.log10(4/(mse_loss.item()))   # img~[-1,1], mse * 1/4
            psnrs.update(psnr)
            
            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if self.args.print_freq > 0 and \
                    (i) % self.args.print_freq == 0:
                self.logging.info('Epoch: [{0}][{1}/{2}]\t'
                      'lr {lr:.3e}\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Total Loss {loss.val:.4f}\t'
                      'MSE Loss {mse_loss.val:.4f}\t'
                      'Psnr {psnr.val:.4f}\t'.format(
                          epoch, i + 1, len(train_loader), lr=lr,
                          batch_time=batch_time, data_time=data_time,
                          loss=losses, mse_loss=mse_losses, psnr=psnrs))  

        self.logging.info('Epoch: {:3d} Total Loss {loss.avg:.4f}  MSE loss {mse_loss.avg:.4f} '
              'Psnr {psnr.avg:.4f}'
              .format(epoch, loss=losses, mse_loss=mse_losses, psnr=psnrs))                  

        return mse_losses.avg, psnrs.avg, lr

    def test(self, val_loader, epoch, dataname='', silence=False, test_ssim=True, args=None, val_data=None):
        batch_time = AverageMeter()
        mse_losses = AverageMeter()
        psnrs = AverageMeter()
        ssims = AverageMeter()


        # switch to evaluate mode
        self.diffusion.adapter.eval()

        start = time.time()
        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
                images = images.float()
                images = images.to(self.device)
                batch_size, _, height, width = images.shape
                # compute outputs
                deep_outputs = self.diffusion.my_sample(images)


                for j in range(images.shape[0]):
                    mse_loss = self.criterion(deep_outputs[j], images[j])
                    # measure error and record loss
                    mse_losses.update(mse_loss.item(), 1)
                    psnr = 10*np.log10(4/(mse_loss.item()))   # img~[-1,1], mse * 1/4
                    psnrs.update(psnr)

                    if test_ssim:
                        ori_image = images[j, 0].cpu().numpy()*255
                        rec_image = deep_outputs[j, 0].cpu().numpy()*255
                        ssim = calculate_ssim(ori_image, rec_image)
                        ssims.update(ssim)

                    if args.is_visualization:
                        img = deep_outputs[j].data.cpu().numpy()
                        img = img[0]
                        save_dir = os.path.join(args.save, 'result', dataname)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        filename = val_data.imgs[i*batch_size+j][0].split('/')[-1].split('.')[0] + '_PSNR_%.2f'%(psnr)+'.png' 
                        save_name = os.path.join(save_dir, filename)
                        cv2.imwrite(save_name, img*255)

                # measure elapsed time
                batch_time.update(time.time() - start)
                start = time.time()

        if not silence:
            self.logging.info(dataname + ' Epoch: {:d}  MSE loss {mse_loss.avg:.4f} Psnr {psnr.avg:.4f} Time {time.avg:.5f}'
                  .format(epoch, mse_loss=mse_losses, psnr=psnrs, time=batch_time))
            if test_ssim:
                self.logging.info('SSIM {ssim.avg:.4f}'.format(ssim=ssims))   
            
                
        return mse_losses.avg, psnrs.avg, ssims.avg







def main():
    args, config = parse_args_and_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.backends.cudnn.benchmark = True

    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    save_file_name = ('_' + args.deg + '_lr_%.3e_' + args.A_type) % (args.lr)
    args.save = os.path.join('/'.join(args.save.split('/')[:-1]), 'DM_' + args.save.split('/')[-1] + save_file_name)

    if not os.path.exists(args.save):
        os.makedirs(args.save)      

    log_name = 'record.log'                                                   
    logging.basicConfig(filename=os.path.join(os.getcwd(), os.path.join(args.save, log_name)),
        format="%(levelname)-10s %(asctime)-20s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
    level=logging.INFO)
    console=logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter=logging.Formatter("%(levelname)-10s %(asctime)-20s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    diffusion = Diffusion(args, config, device)


    if args.adap_pth != "empty":
        # diffusion.adapter.load_pth(args.adap_pth, device)
        logging.info("=> loading adapter diffusion : %s", args.adap_pth)
    else:
        logging.info("=> creating adapter diffusion")

    
    
    _, train_dataloader, val_dataset, val_dataloader = get_data(args)
    
    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(diffusion.adapter.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(train_dataloader), epochs=args.epochs, three_phase=True)
    trainer = Trainer(diffusion, criterion, optimizer, scheduler, args, logging, device)

    

    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value) 

    if args.is_test:
        logging.info('----------------')  
        val_mse, val_psnr, val_ssim = trainer.test(val_dataloader, 1, dataname = 'face', test_ssim=True, args=args, val_data=val_dataset)

        logging.info('----------------') 
        
        return   


    # train and val
    for epoch in range(args.re_epoch+1, args.epochs+1):
        torch.manual_seed(args.seed+epoch)

        # train for one epoch
        trainer.train(train_dataloader, epoch, args=args)        
        

        logging.info('----------------')
        val_mse, val_psnr, val_ssim = trainer.test(val_dataloader, epoch, dataname = 'face', test_ssim=True, args=args, val_data=val_dataset)
        if epoch > 95:
            diffusion.sample(args.simplified)
        
        logging.info('----------------')
    
        save_checkpoint({
                'args': args,
                'epoch': epoch,
                'state_dict': diffusion.adapter.state_dict(),
            }, epoch, val_mse, val_psnr, val_ssim, args.save)



if __name__ == '__main__':
    main()

