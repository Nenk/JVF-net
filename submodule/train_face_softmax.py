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
from option import Resnet_config
from torch.utils.data import DataLoader
from dataset import  RAVDESS_face_Dataset
from model.model4 import ResNet

from utils import util
from utils.util import Logger, print_log
from test_face import validate_for_triplet

from pytorch_metric_learning import losses, miners, distances, reducers, samplers, trainers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


timestamp = time.strftime("%Y-%m-%d-%H:%M")

class trainer():
    def __init__(self):
        # --------------------hyperparameters & data loaders-------------------- #
        self.cfg = Resnet_config.opt_softmax

        # train_list, test_list, actor_num, emotion_num = csv_to_list(self.cfg['csv_list'], 0.1)
        self.train_cfg = self.cfg['training']
        self.net_cfg = self.cfg['network']
        self.data_root = self.cfg['data_root']
        self.data_cfg = self.cfg['data_cfg']

        self.save_path = os.path.join(self.cfg['save_path'], time.strftime("%Y-%h-%d:%H:%M"))

        batch_size = self.train_cfg['batch_size']
        num_workers = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.face_train_dataset = RAVDESS_face_Dataset(self.data_root, self.data_cfg, split='train')
        self.face_test_dataset = RAVDESS_face_Dataset(self.data_root, self.data_cfg, split='valid')
        # self.sampler = samplers.MPerClassSampler(self.face_train_dataset.label, m=5, batch_size=batch_size,
        #                                          length_before_new_iter=len(self.face_train_dataset))

        self.train_loader = DataLoader(self.face_train_dataset, batch_size=batch_size, drop_last=False,
                                       shuffle=False, num_workers=num_workers, pin_memory=True)

        self.val_loader = DataLoader(self.face_test_dataset, batch_size=batch_size, drop_last=False,
                                     shuffle=True, num_workers=num_workers, pin_memory=True)

        # print("Train_loader numbers: {0:}, val_loader numbers: {1:}".format(len(train_list), len(test_list)))
        self.net_cfg['class_num'] = len(self.face_train_dataset.emo_info)

        # --------------------model & loss & optimizer----------------------- #
        if self.net_cfg['type'] == 'resnet':
            self.model = ResNet(class_num=self.net_cfg['class_num'], include_top=False)
            # utils.load_state_dict(model, weight_file)
            # model.fc.reset_parameters()
        self.model = self.model.to(self.device)

        ### cross entropy loss
        self.loss_func  = nn.CrossEntropyLoss()

        self.loss_func = self.loss_func.to(self.device)

        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.train_cfg['lr'])

        self.lr_scheduler = None

        # -----------------------log and print ------------------------------ #
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.logger = print_log(self.save_path, 'resnet-softmax')

        self.logger.write(str(self.net_cfg))
        self.logger.write(str(self.train_cfg))

        self.epoch_time = util.AverageMeter()
        self.batch_time = util.AverageMeter()
        self.loss = util.AverageMeter()
        self.top1 = util.AverageMeter()

        self.last_iteration = 0
        # best_top1 = 0
        self.print_freq = 1
        self.start_epoch  = 1
        self.max_epoch = self.train_cfg['max_epoch']

    # --------------------train & validation ---------------------------- #
    def train(self):
        print("Start training")

        for epoch in range(self.start_epoch, self.max_epoch+1):  # 调整进度条宽度为80

            self.model.train()
            self.optim.zero_grad()

            epoch_end = time.time()
            iter_end = time.time()

            for batch_idx, (imgs, target) in enumerate(self.train_loader,start=1):
                iteration = batch_idx + epoch * len(self.train_loader)

                # self.iteration = iteration
                imgs, target = imgs.to(self.device), target.to(self.device)
                target = target-1
                embeddings = self.model(imgs)

                loss = self.loss_func(embeddings,target)

                ## measure accuracy for softmax
                pred = embeddings.argmax(dim=1, keepdim=False)
                correct = pred.eq(target.view_as(pred)).sum().item() / float(imgs.size(0))  # tensor --> python number

                # ----------------------------------------
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()  # update lr

                # ---------迭次信息输出----------------- #
                self.batch_time.update(time.time() - iter_end)
                self.top1.update(correct, imgs.size(0))
                self.loss.update(loss.item(), imgs.size(0))
                # top1.update(correct, imgs.size(0))
                if iteration % self.print_freq == 0:
                    self.logger.write('Train:[{0}/{1}]\t '
                                      'epoch:{2} iter:{3}, Batch Time:{4:.2f}\t' \
                                      'Loss: {5:.4f}, accuracy:{6:.4f}\t' \
                                      'lr {7:.6f}'.format(\
                                batch_idx, len(self.train_loader), epoch, iteration,
                                self.batch_time.val, self.loss.val, self.top1.val,
                                self.optim.param_groups[0]['lr'] ))
                iter_end = time.time()

            # -------------------轮次信息输出--------------------- #
            self.epoch_time.update(time.time() - epoch_end)

            self.logger.write('\n Epoch_summary: epoch: {0}\t ' \
                              'iter: {1}, Epoch Time: {2:.2f}\t' \
                              'Loss: {3:.4f}, accuracy:{4:.4f}, lr {5:.6f}\t'  \
                         .format(\
                         epoch, iteration, self.epoch_time.val,
                         self.loss.val,  self.top1.val,
                         self.optim.param_groups[0]['lr']))

            # ----------------保存模型----------------------------------------#
            self.checkpoint_file = os.path.join(self.save_path,'{}-checkpoint-{}.pth'.
                                                format(self.net_cfg['type'], timestamp))

            torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'arch': self.model.__class__.__name__,  # class name
                # 'optim_state_dict': optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'loss': self.loss,
            }, self.checkpoint_file)


            if epoch % 20 == 0:
                best_file = os.path.join(self.save_path,'{}-epoch-{}.pth'.
                                         format(self.net_cfg['type'], epoch))
                shutil.copy(self.checkpoint_file, best_file)

            self.loss.reset()
            self.top1.reset()

    def resume_training(self, iteration, device):
        if self.last_iteration != 0 and (iteration - 1) != self.last_iteration:
            pass

    def validate(self, test_loader, model, device):
        correct = 0
        total = 0

        # Validate
        model.eval()
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                # targets = targets.squeeze()
                outputs = model(inputs)

                _, predicted = outputs.max(1)
                batch_size = targets.size(0)
                total += batch_size
                correct += predicted.eq(targets).sum().item()

                print("batch_ix [{:4d}/{:4d}] test acc: {:6.3f}% ".
                      format(batch_index, len(test_loader), 100.0 * correct / total))

            print("average test acc: {:6.3f}% ".format(100.0 * correct / total))

if __name__ == '__main__':
    Trainer = trainer()
    Trainer.train()

    # checkpoint_file = "/home/fz/2-VF-feature/JVF-net/saved/Resnet/2021-May-26:18:04/resnet-epoch-60.pth"
    # Trainer.validate(checkpoint_file)
