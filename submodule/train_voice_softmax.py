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
from option import PASE_config
from torch.utils.data import DataLoader
from dataset import RAVDESS_face_Dataset, RAVDESS_voice_Dataset
from model.SVHF import AudioStream, ResNet, SVHFNet
from pase.models.frontend import wf_builder

from utils import util
from utils.util import Logger, print_log
from test_face import validate_for_triplet
from tensorboardX import SummaryWriter


from pytorch_metric_learning import losses, miners, distances, reducers, samplers, trainers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


timestamp = time.strftime("%Y-%m-%d-%H:%M")

class trainer():
    def __init__(self):
        # --------------------hyperparameters & data loaders-------------------- #
        self.cfg = PASE_config.opt_softmax
        self.train_cfg = self.cfg['training']
        self.net_cfg = self.cfg['network']
        self.data_cfg = self.cfg['data_cfg']
        self.data_root = self.cfg['data_root']
        self.save_path = os.path.join(self.cfg['save_path'], time.strftime("%Y-%h-%d:%H:%M"))

        batch_size = self.train_cfg['batch_size']
        num_workers = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.voice_train_dataset = RAVDESS_voice_Dataset(self.data_root, self.data_cfg, split='train')
        self.voice_test_dataset = RAVDESS_voice_Dataset(self.data_root, self.data_cfg, split='valid')
        # self.sampler = samplers.MPerClassSampler(self.face_train_dataset.label, m=5, batch_size=batch_size,
        #                                          length_before_new_iter=len(self.face_train_dataset))

        self.train_loader = DataLoader(self.voice_train_dataset, batch_size=batch_size, drop_last=False,
                                       shuffle=True, num_workers=num_workers, pin_memory=True)

        self.val_loader = DataLoader(self.voice_test_dataset, batch_size=batch_size, drop_last=False,
                                     shuffle=True, num_workers=num_workers, pin_memory=True)

        # print("Train_loader numbers: {0:}, val_loader numbers: {1:}".format(len(train_list), len(test_list)))
        self.net_cfg['class_num'] = len(self.voice_train_dataset.emo_info)

        # --------------------model & loss & optimizer----------------------- #
        if self.net_cfg['type'] == 'pase':
            # utils.load_state_dict(model, weight_file)
            # model.fc.reset_parameters()
            pase = wf_builder(self.cfg['pase_cfg_path'])  # read pre-trained model
            pase.load_pretrained(self.cfg['pase_ckpt_path'], load_last=True, verbose=True)
            self.aud_stream = AudioStream(pase)

        self.model = self.aud_stream.to(self.device)

        ### cross entropy loss
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func = self.loss_func.to(self.device)
        # self.optim = torch.optim.SGD(self.model.parameters(), lr=self.train_cfg['lr'],
        #                              momentum=self.train_cfg['momentum'] )
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg['lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1000,
                                                            gamma=0.5, last_epoch = -1)
        # self.lr_scheduler = None

        # -----------------------log and print ------------------------------ #
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.logger = print_log(self.save_path, 'resnet-softmax')
        self.writer = SummaryWriter(self.save_path)
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

            for batch_idx, (voices, target) in enumerate(self.train_loader, start=1):
                iteration = batch_idx + epoch * len(self.train_loader)
                target = target - 1
                # self.iteration = iteration
                voices, target = voices.to(self.device), target.to(self.device)

                embeddings = self.model(voices)
                loss = self.loss_func(embeddings, target)

                # measure accuracy for softmax
                pred = embeddings.argmax(dim=1, keepdim=False)
                correct = pred.eq(target.view_as(pred)).sum().item() / float(voices.size(0))  # tensor --> python number

                # ----------------------------------------
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.lr_scheduler is not None:
                   self.lr_scheduler.step()  # update lr
                # ---------迭次信息输出----------------- #
                self.batch_time.update(time.time() - iter_end)
                self.top1.update(correct, voices.size(0))
                self.loss.update(loss.item(), voices.size(0))
                # top1.update(correct, imgs.size(0))
                if iteration % self.print_freq == 0:
                    self.logger.write('Train:[{0}/{1}]\t '
                                      'epoch:{2} iter:{3}, Batch Time:{4:.2f}\t' \
                                      'Loss: {5:.4f}, accuracy:{6:.6f}\t' \
                                      'lr {7:.6f}'.format(\
                                batch_idx, len(self.train_loader), epoch, iteration,
                                self.batch_time.val, self.loss.val, self.top1.val,
                                self.optim.param_groups[0]['lr'] ))
                iter_end = time.time()


            # -------------------轮次信息输出--------------------- #
            self.epoch_time.update(time.time() - epoch_end)

            self.logger.write('\nEpoch_summary: epoch: {0}\t ' \
                              'iter: {1}, Epoch Time: {2:.2f}\t' \
                              'Loss_avg: {3:.4f}, accuracy:{4:.6f}, lr {5:.6f}\t \n'  \
                         .format(\
                         epoch, iteration, self.epoch_time.val,
                         self.loss.avg,  self.top1.avg,
                         self.optim.param_groups[0]['lr']))
            self.writer.add_scalar('train/{}_loss'.format('total'), loss.item(), global_step=epoch)
            self.writer.add_scalar('train/{}_acc'.format('total'), correct, global_step=epoch)
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
