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
from dataset import VGG_Face_Dataset, RAVDESS_Face_Dataset
from model.model4 import ResNet
from utils.parse_dataset import csv_to_list
from utils import utils
from test_face import validate_for_triplet

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
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
        batch_size=4,
        max_epoch=3000,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        gamma=0.9,  # "lr_policy: step"
        step_size=1000,  # "lr_policy: step"
        interval_validate=1000,
    ),
    'csv_list': '../dataset/RAVDESS/RAVDESS_image.csv',
    'log_dir': '../log/',
    'checkpoint_dir': '../saved',
}

if __name__ == '__main__':
    # --------------------hyperparameters & data loaders-------------------- #
    train_list, test_list, actor_num, emotion_num = csv_to_list(configure['csv_list'], 0.2)

    train_cfg = configure['training']
    net_cfg = configure['network']
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    face_train_dataset = RAVDESS_Face_Dataset(train_list)
    face_test_dataset = RAVDESS_Face_Dataset(test_list)
    train_loader = DataLoader(face_train_dataset, batch_size=train_cfg['batch_size'], drop_last=False,
                             shuffle=True, num_workers=0, pin_memory=True)

    val_loader = DataLoader(face_test_dataset, batch_size=train_cfg['batch_size'], drop_last=False,
                             shuffle=True, num_workers=0, pin_memory=True)

    print("Train_loader numbers: {0:}, val_loader numbers: {1:}".format(len(train_list), len(test_list)))
    net_cfg['class_num'] = emotion_num

    #  ------------------model & loss & optimizer---------------------  #
    if net_cfg['type'] == 'resnet':
        model = ResNet(class_num=net_cfg['class_num'], include_top=False)
        # utils.load_state_dict(model, weight_file)
        # model.fc.reset_parameters()
    model = model.to(device)

    ### cross entropy loss
    cross_Entropy_loss = nn.CrossEntropyLoss()
    cross_Entropy_loss = cross_Entropy_loss.to(device)
    ### triplet loss setting
    distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, reducer=reducer, distance=distance)
    mining_func = miners.TripletMarginMiner(margin=0.2, type_of_triplets="smeihard", distance=distance)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    optim = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])

    lr_scheduler = None

    # -----------------------log and print ------------------------------ #
    log = utils.print_log(configure['log_dir'], [net_cfg['type'], timestamp])
    log.write(str(net_cfg))
    log.write(str(train_cfg))
    epoch_time = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # --------------------train & validation ---------------------------- #
    last_iteration = 0
    epoch = 0
    best_top1 = 0
    print_freq = 1
    max_epoch = train_cfg['max_epoch']
    print("Start training")
    for epoch in tqdm.trange(epoch, max_epoch, desc='Train', ncols=80):  # 调整进度条宽度为80
        # self.epoch = epoch
        # top5 = utils.AverageMeter()
        model.train()
        optim.zero_grad()

        epoch_end = time.time()
        iter_end = time.time()

        for batch_idx, (imgs, target) in enumerate(train_loader):
            iteration = batch_idx + epoch * len(train_loader)

            if last_iteration != 0 and (iteration - 1) != last_iteration:
                continue  # for resuming model
            # self.iteration = iteration

            imgs, target = imgs.cuda(), target.cuda()
            embeddings = model(imgs)
            indices_tuple = mining_func(embeddings, target)
            loss = loss_func(embeddings, target, indices_tuple)
            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            ## measure accuracy for softmax
            # pred = embeddings.argmax(dim=1, keepdim=False)
            # correct = pred.eq(target.view_as(pred)).sum().item() / float(imgs.size(0))  # tensor --> python number
            # prec1 = utils.accuracy(output.data, target.data, topk=[1])

            # ----------------------------------------
            optim.zero_grad()
            loss.backward()
            optim.step()
            if lr_scheduler is not None:
                lr_scheduler.step()  # update lr

            if (iteration + 1) % 50 == 0:
                if val_loader:
                    validate_for_triplet(face_train_dataset, face_test_dataset, model, accuracy_calculator)

            # ---------迭次信息输出----------------- #
            batch_time.update(time.time() - iter_end)
            iter_end = time.time()
            losses.update(loss.item(), imgs.size(0))
            # top1.update(correct, imgs.size(0))
            if iteration % print_freq == 0:
                log_str = 'Train:[{0}/{1}]\t epoch:{epoch:} iter:{iteration:}\t' \
                          'Batch Time: {batch_time.val:.3f}\t' \
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                          'lr {lr:.6f}'.format( \
                    batch_idx, len(train_loader), epoch=epoch, iteration=iteration,
                    lr=optim.param_groups[0]['lr'], batch_time=batch_time,
                    loss=losses)
                print(log_str)
                log.write(log_str)


        # -------------------轮次信息输出--------------------- #
        epoch_time.update(time.time() - epoch_end)
        log_str = '\n Epoch_summary: [{0}/{1}]\t epoch: {epoch:}\t ' \
                  'iter: {iteration:}\t' \
                  'Epoch Time: {epoch_time.val:.3f} Loss: {loss.avg:.4f}\t' \
                  'lr {lr:.6f}' .format( \
             batch_idx, len(train_loader), epoch=epoch,
             iteration=iteration, epoch_time=epoch_time, loss=losses,
             lr=optim.param_groups[0]['lr'], )

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
            if None:
                best_file = os.path.join(configure['checkpoint_dir'],
                                         '{}-model_best-{}.pth'.format(net_cfg['type'], timestamp))
                shutil.copy(checkpoint_file, best_file)

        losses.reset()
        top1.reset()
