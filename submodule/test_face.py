from __future__ import division

# from models import *
# from utils.utils import *
# from utils.datasets import *
# from utils.augmentations import *
# from utils.transforms import *
# from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def Tester(object):
    def __init__(self, cmd, cuda, model, criterion, optimizer,
                 train_loader, val_loader, log_file, max_epoch,
                 interval_validate=None, lr_scheduler=None,
                 checkpoint_dir=None, print_freq=1):
        self.cmd = cmd
        self.cuda = cuda

        self.model = model
        self.criterion = criterion
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = datetime.datetime.now()

        if cmd == 'train':
            self.interval_validate = len(self.train_loader) if interval_validate is None else interval_validate

        self.epoch = 0
        self.max_epoch = max_epoch

        self.iteration = 0

        # self.max_iter = max_iter
        self.best_top1 = 0
        self.best_top5 = 0
        self.print_freq = print_freq

        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file

    def validate(self):
        batch_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        training = self.model.training
        self.model.eval()

        end = time.time()
        for batch_idx, (imgs, target, img_files, class_ids) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration={} epoch={}'.format(self.iteration, self.epoch), ncols=80, leave=False):

            gc.collect()
            if self.cuda:
                imgs, target = imgs.cuda(), target.cuda(async=True)
            imgs = Variable(imgs, volatile=True)
            target = Variable(target, volatile=True)

            output = self.model(imgs)
            loss = self.criterion(output, target)

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while validating')

            # measure accuracy and record loss
            prec1, prec5 = utils.accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(prec1[0], imgs.size(0))
            top5.update(prec5[0], imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % self.print_freq == 0:
                log_str = 'Test: [{0}/{1}/{top1.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                          'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t' \
                          'Prec@5: {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
                    batch_time=batch_time, loss=losses, top1=top1, top5=top5)
                print(log_str)
                self.print_log(log_str)
        if self.cmd == 'test':
            log_str = 'Test: [{0}/{1}/{top1.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t' \
                      'Prec@5: {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
                batch_time=batch_time, loss=losses, top1=top1, top5=top5)
            print(log_str)
            self.print_log(log_str)

        if self.cmd == 'train':
            is_best = top1.avg > self.best_top1
            self.best_top1 = max(top1.avg, self.best_top1)
            self.best_top5 = max(top5.avg, self.best_top5)

            log_str = 'Test_summary: [{0}/{1}/{top1.count:}] epoch: {epoch:} iter: {iteration:}\t' \
                      'BestPrec@1: {best_top1:.3f}\tBestPrec@5: {best_top5:.3f}\t' \
                      'Time: {batch_time.avg:.3f}\tLoss: {loss.avg:.4f}\t' \
                      'Prec@1: {top1.avg:.3f}\tPrec@5: {top5.avg:.3f}\t'.format(
                batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
                best_top1=self.best_top1, best_top5=self.best_top5,
                batch_time=batch_time, loss=losses, top1=top1, top5=top5)
            print(log_str)
            self.print_log(log_str)

            checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pth.tar')
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'best_top1': self.best_top1,
                'batch_time': batch_time,
                'losses': losses,
                'top1': top1,
                'top5': top5,
            }, checkpoint_file)
            if is_best:
                shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            if (self.epoch + 1) % 10 == 0:  # save each 10 epoch
                shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'checkpoint-{}.pth.tar'.format(self.epoch)))

            if training:
                self.model.train()