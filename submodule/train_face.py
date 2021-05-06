# coding=utf-8
import time
import tqdm
import gc
import os
import shutil
import torch
import numpy as np
import torch.nn as nn

# import model.model4 as model4
from torch.utils.data import DataLoader
from dataset import VGG_Face_Dataset
from model.model4 import ResNet
from utils.parse_dataset import csv_to_list
from utils import utils
from pytorch_metric_learning import losses

# import models.resnet as ResNet
# import models.senet as SENet
# import models.vgg as VGG
# from face_image_embeddding.utils import utils, face_dataset
# from trainer import Trainer

timestamp = time.strftime("%Y-%m-%d,%H,%M")

configure = {
    'network': dict(
        type='resnet',
        class_num=24),

    'training': dict(
        start_epoch=0,
        start_iteration=0,
        batch_size=32,
        max_epoch=3000,
        lr=0.03,
        momentum=0.9,
        weight_decay=0.0001,
        gamma=0.9,  # "lr_policy: step"
        step_size=1000,  # "lr_policy: step"
        interval_validate=1000,
    ),
    'csv_list': '/home/fz/2-VF-feature/JVF-net/dataset/voclexb-VGG_face-datasets/face_list.csv',
    'log_dir': '../log/',
    'checkpoint_dir': '../saved',
}

if __name__ == '__main__':
    # --------------------hyperparameters & data loaders-------------------- #
    attribution = "identity"  # identity, emotion 类型
    face_list, actor_num = csv_to_list(configure['csv_list'])

    train_cfg = configure['training']
    net_cfg = configure['network']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    face_dataset = VGG_Face_Dataset(face_list, 'train')
    face_loader = DataLoader(face_dataset, batch_size=train_cfg['batch_size'], drop_last=False,
                             shuffle=True, num_workers=0, pin_memory=True)

    val_loader = None
    interval_validate = len(face_loader)  # 验证间隔
    net_cfg['class_num'] = actor_num

    #  ------------------model & loss & optimizer---------------------  #
    if net_cfg['type'] == 'resnet':
        model = ResNet(class_num=net_cfg['class_num'], include_top=False)
        # utils.load_state_dict(model, weight_file)
        # model.fc.reset_parameters()
    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=train_cfg['lr'],
                            momentum=train_cfg['momentum'],
                            weight_decay=train_cfg['weight_decay'])

    Cross_Entropy_loss = nn.CrossEntropyLoss()
    Cross_Entropy_loss = Cross_Entropy_loss.to(device)
    Triplet_loss = losses.TripletMarginLoss()
    Triplet_loss = Triplet_loss.to(device)
    loss_func = Triplet_loss

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, train_cfg['step_size'], gamma=train_cfg['gamma'], last_epoch=-1)
    lr_scheduler = None
    log = utils.print_log(configure['log_dir'], [net_cfg['type'], timestamp])
    log.write(str(net_cfg))
    log.write(str(train_cfg))
    epoch_time = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # --------------------train & validation & save checkpoint---------------- #
    epoch = 0
    last_iteration = 0
    print_freq = 1
    best_top1 = 0
    max_epoch = train_cfg['max_epoch']
    print("train max epoch {0}".format(max_epoch))
    for epoch in tqdm.trange(epoch, max_epoch, desc='Train', ncols=80):  # 调整进度条宽度为80
        # self.epoch = epoch
        # top5 = utils.AverageMeter()
        model.train()
        optim.zero_grad()

        epoch_end = time.time()
        iter_end = time.time()

        for batch_idx, (imgs, target) in enumerate(face_loader):
            iteration = batch_idx + epoch * len(face_loader)
            # print('\n{} {} {}'.format(batch_idx, epoch, len(self.train_loader)))
            data_time.update(time.time() - iter_end)

            # gc.collect()  # 清理内存
            if last_iteration != 0 and (iteration - 1) != last_iteration:
                continue  # for resuming model
            # self.iteration = iteration

            imgs, target = imgs.cuda(), target.cuda()
            output = model(imgs)
            loss = loss_func(output, target)
            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            # measure accuracy and record loss
            pred = output.argmax(dim=1, keepdim=False)
            correct = pred.eq(target.view_as(pred)).sum().item() / train_cfg['batch_size']  # tensor --> python number
            # prec1 = utils.accuracy(output.data, target.data, topk=[1])
            # ----------------------------------------

            losses.update(loss.item(), imgs.size(0))
            top1.update(correct, imgs.size(0))

            optim.zero_grad()
            loss.backward()
            optim.step()
            if lr_scheduler is not None:
                lr_scheduler.step()  # update lr

            # ---------迭次信息输出----------------- #
            batch_time.update(time.time() - iter_end)
            iter_end = time.time()
            if iteration % print_freq == 0:
                log_str = 'Train: [{0}/{1}]\t epoch: {epoch:}\t iter: {iteration:}\t' \
                          'Time: {batch_time.val:.3f}  Data: {data_time.val:.3f}\t' \
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                          'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t' \
                          'lr {lr:.6f}'.format(
                    batch_idx, len(face_loader), epoch=epoch, iteration=iteration,
                    lr=optim.param_groups[0]['lr'], batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1)
                print(log_str)
                log.write(log_str)

        is_best = top1.avg > best_top1
        best_top1 = max(top1.avg, best_top1)

        # -------------------轮次信息输出--------------------- #
        epoch_time.update(time.time() - epoch_end)
        log_str = '\n Train_epoch_summary: [{0}/{1}/{top1.count:}]\t epoch: {epoch:}\t iter: {iteration:}\t' \
                  'Epoch Time: {epoch_time.val:.3f} Loss: {loss.avg:.4f}\t' \
                  'Prec@1: {top1.avg:.3f}  BestPrec@1:{best_top1:.3F} lr {lr:.6f}\t' \
            .format( batch_idx, len(face_loader), epoch=epoch, iteration=iteration,
                     epoch_time=epoch_time, loss=losses,
                     top1=top1, best_top1=best_top1, lr=optim.param_groups[0]['lr'], )

        print(log_str)
        log.write(log_str)

        # ----------------保存模型----------------------------------------#
        if epoch % 5 == 0 or epoch == 1:
            checkpoint_file = os.path.join(configure['checkpoint_dir'],
                                           '{}-checkpoint-{}.pth'.format(net_cfg['type'],
                                                                         time.strftime("%Y-%m-%d,%H,%M")))
            torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'arch': model.__class__.__name__,  # class name
                # 'optim_state_dict': optim.state_dict(),
                'model_state_dict': model.state_dict(),
                'best_top1': best_top1,
                'losses': losses,
                'top1': top1,
            }, checkpoint_file)
            if is_best:
                best_file = os.path.join(configure['checkpoint_dir'],
                                         '{}-model_best-{}.pth'.format(net_cfg['type'], timestamp))
                shutil.copy(checkpoint_file, best_file)

        losses.reset()
        top1.reset()
