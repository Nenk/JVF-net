# coding=utf-8
import time
import tqdm
import gc
import os
import shutil
import torch
import numpy as np
import torch.nn as nn
import pytorch_metric_learning
# import model.model4 as model4
from torch.utils.data import DataLoader
from dataset import VGG_Face_Dataset, RAVDESS_Face_Dataset
from model.model4 import ResNet
from utils.parse_dataset import csv_to_list
from utils import util, config
from utils.util import Logger, print_log
from test_face import validate_for_triplet

from pytorch_metric_learning import losses, miners, distances, reducers, samplers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


timestamp = time.strftime("%Y-%m-%d-%H:%M")

class trainer():
    def __init__(self):
        # --------------------hyperparameters & data loaders-------------------- #
        self.cfg = config.configure

        train_list, test_list, actor_num, emotion_num = csv_to_list(self.cfg['csv_list'], 0.2)
        self.train_cfg = self.cfg['training']
        self.net_cfg = self.cfg['network']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.face_train_dataset = RAVDESS_Face_Dataset(train_list)
        self.face_test_dataset = RAVDESS_Face_Dataset(test_list)
        self.sampler = samplers.MPerClassSampler(self.face_train_dataset.label, m=4, batch_size=None,
                                                 length_before_new_iter=len(self.face_train_dataset))

        self.train_loader = DataLoader(self.face_train_dataset, batch_size=self.train_cfg['batch_size'], drop_last=False,
                                       sampler=self.sampler, shuffle=False, num_workers=0, pin_memory=True)

        self.val_loader = DataLoader(self.face_test_dataset, batch_size=self.train_cfg['batch_size'], drop_last=False,
                                shuffle=True, num_workers=0, pin_memory=True)

        print("Train_loader numbers: {0:}, val_loader numbers: {1:}".format(len(train_list), len(test_list)))
        self.net_cfg['class_num'] = emotion_num

        #  ------------------model & loss & optimizer---------------------  #
        if self.net_cfg['type'] == 'resnet':
            self.model = ResNet(class_num=self.net_cfg['class_num'], include_top=False)
            # utils.load_state_dict(model, weight_file)
            # model.fc.reset_parameters()
        self.model = self.model.to(device)

        ### cross entropy loss
        cross_Entropy_loss = nn.CrossEntropyLoss()
        cross_Entropy_loss = cross_Entropy_loss.to(device)
        ### triplet loss setting
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)

        self.loss_func = losses.TripletMarginLoss(margin=0.2, reducer=reducer, distance=distance)
        self.mining_func = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard", distance=distance)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg['lr'])

        self.lr_scheduler = None

        # -----------------------log and print ------------------------------ #
        self.logger = print_log(self.cfg['log_dir'], time.strftime("%Y-%m-%d,%H,%M"))

        self.logger.write(str(self.net_cfg))
        self.logger.write(str(self.train_cfg))

        self.epoch_time = util.AverageMeter()
        self.batch_time = util.AverageMeter()
        self.loss = util.AverageMeter()
        self.top1 = util.AverageMeter()

        self.last_iteration = 0
        # best_top1 = 0
        self.print_freq = 1
        self.start_epoch  = 0
        self.max_epoch = self.train_cfg['max_epoch']

    # --------------------train & validation ---------------------------- #
    def train(self):
        print("Start training")

        for epoch in range(self.start_epoch, self.max_epoch):  # 调整进度条宽度为80

            self.model.train()
            self.optim.zero_grad()

            epoch_end = time.time()
            iter_end = time.time()

            for batch_idx, (imgs, target) in enumerate(self.train_loader):
                iteration = batch_idx + epoch * len(self.train_loader)

                # self.iteration = iteration
                imgs, target = imgs.cuda(), target.cuda()
                embeddings = self.model(imgs)
                indices_tuple = self.mining_func(embeddings, target)
                loss = self.loss_func(embeddings, target, indices_tuple)
                if np.isnan(float(loss.item())):
                    raise ValueError('loss is nan while training')

                ## measure accuracy for softmax
                # pred = embeddings.argmax(dim=1, keepdim=False)
                # correct = pred.eq(target.view_as(pred)).sum().item() / float(imgs.size(0))  # tensor --> python number
                # prec1 = utils.accuracy(output.data, target.data, topk=[1])

                # ----------------------------------------
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()  # update lr

                # ---------迭次信息输出----------------- #
                self.batch_time.update(time.time() - iter_end)
                self.loss.update(loss.item(), imgs.size(0))
                # top1.update(correct, imgs.size(0))
                if iteration % self.print_freq == 0:
                    log_str = 'Train:[{0}/{1}]\t epoch:{epoch:} iter:{iteration:}\t' \
                              'Batch Time: {batch_time.val:.3f}\t' \
                              'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                              'lr {lr:.6f}'.format( \
                        batch_idx, len(self.train_loader), epoch=epoch, iteration=iteration,
                        lr=self.optim.param_groups[0]['lr'], batch_time=self.batch_time,
                        loss=self.loss)

                    self.logger.write(log_str)
                iter_end = time.time()

            # -------------------轮次信息输出--------------------- #
            self.epoch_time.update(time.time() - epoch_end)
            log_str = '\n Epoch_summary: [{0}/{1}]\t epoch: {epoch:}\t ' \
                      'iter: {iteration:}\t' \
                      'Epoch Time: {epoch_time.val:.3f} Loss: {loss.avg:.4f}\t' \
                      'lr {lr:.6f}' .format( \
                 batch_idx, len(self.train_loader), epoch=epoch,
                 iteration=iteration, epoch_time=self.epoch_time, loss=self.loss,
                 lr=self.optim.param_groups[0]['lr'], )

            self.logger.write(log_str)

            # ----------------保存模型----------------------------------------#
            self.checkpoint_file = os.path.join(self.cfg['checkpoint_dir'],'{}-checkpoint-{}.pth'.
                                                format(self.net_cfg['type'], timestamp))
            torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'arch': self.model.__class__.__name__,  # class name
                # 'optim_state_dict': optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'loss': self.loss,
            }, self.checkpoint_file)

            self.validate(self.checkpoint_file)
            if epoch % 20 == 0:
                best_file = os.path.join(self.cfg['checkpoint_dir'],'{}-model-epoch-{}-T-{}.pth'.
                                         format(self.net_cfg['type'], epoch, time.strftime("%Y-%m-%d-%H:%M")))
                shutil.copy(self.checkpoint_file, best_file)

            self.loss.reset()

    def resume_training(self, iteration, device):
        if self.last_iteration != 0 and (iteration - 1) != self.last_iteration:
            pass

    def validate(self, checkpoint_file):
        print("start validate")
        torch.cuda.empty_cache()
        checkpoint = torch.load(checkpoint_file)
        ckpt = checkpoint['model_state_dict']
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1, avg_of_avgs=False)
        self.validate_for_triplet = validate_for_triplet(self.model, accuracy_calculator, batch_size = 24)
        with torch.no_grad():
            # self.model.eval()
            self.validate_for_triplet.get_accuracy(self.face_train_dataset, self.face_test_dataset)

if __name__ == '__main__':
    Trainer = trainer()
    Trainer.train()
    checkpoint_file = "/home/fz/2-VF-feature/JVF-net/saved/resnet-model-epoch-20-T-2021-05-12-14:56.pth"

    Trainer.validate(checkpoint_file)
