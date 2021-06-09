import torch
import torch.nn as nn

import os
import time
import shutil
import yaml
from torch.utils.data import DataLoader
from dataset import RAVDESS_voice_Dataset, RAVDESS_face_Dataset, custom_collate_fn
from utils.util import Logger, print_log, AverageMeter, cycle, Saver
from utils.JVF_Miner import TripletMarginMiner
from tensorboardX import SummaryWriter

from model.SVHF import AudioStream, ResNet, SVHFNet

from test import validate_for_VF_triplet, inference
from pytorch_metric_learning import losses, miners, distances, reducers, samplers, trainers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator



class trainer(object):
    def __init__(self, config):

        self.opt = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 'cuda' if torch.cuda.is_available() else 'cpu'
        # load_raw = True if config['model'] == 'model4' or config['model'] == 'model5' else False
        self.face_data_root = self.opt['face_data_root']
        self.face_data_cfg = self.opt['face_data_cfg']
        self.face_train_data = RAVDESS_face_Dataset(self.face_data_root, self.face_data_cfg, split = 'train')
        self.face_val_data = RAVDESS_face_Dataset(self.face_data_root, self.face_data_cfg, split = 'valid')

        self.voice_data_root = self.opt['voice_data_root']
        self.voice_data_cfg = self.opt['voice_data_cfg']
        self.voice_train_data = RAVDESS_voice_Dataset(self.voice_data_root, self.voice_data_cfg, split = 'train')
        self.voice_val_data = RAVDESS_voice_Dataset(self.voice_data_root, self.voice_data_cfg, split = 'valid')

        self.voice_batch_size = self.opt['voice_batch_size']
        self.face_batch_size = self.opt['face_batch_size']
        self.num_workers = self.opt['num_workers']

        self.voice_sampler = samplers.MPerClassSampler(self.voice_train_data.label, m=3, batch_size=self.voice_batch_size,
                                                 length_before_new_iter=len(self.voice_train_data))
        self.face_sampler = samplers.MPerClassSampler(self.face_train_data.label, m=4, batch_size=self.face_batch_size,
                                                 length_before_new_iter=len(self.face_train_data))
        self.voice_train_loader = DataLoader(self.voice_train_data,
                                             batch_size=self.voice_batch_size,
                                             sampler=self.voice_sampler,
                                             num_workers= self.num_workers, shuffle=False, drop_last=True)
        self.voice_val_loader = DataLoader(self.voice_val_data,
                                           batch_size=self.voice_batch_size,
                                           num_workers=self.num_workers, shuffle=True, drop_last=True)
        self.voice_test_loader = self.voice_val_loader

        self.face_train_loader = DataLoader(self.face_train_data,
                                            batch_size=self.face_batch_size,
                                            sampler=self.face_sampler,
                                            num_workers=self.num_workers, shuffle=False, drop_last=True)
        self.face_val_loader = DataLoader(self.face_val_data,
                                          batch_size=self.face_batch_size,
                                          num_workers=self.num_workers, shuffle=True, drop_last=True)
        self.face_test_loader = self.face_val_loader

        self.voice_train_iterator = iter(cycle(self.voice_train_loader))
        self.voice_val_iterator = iter(cycle(self.voice_val_loader))
        self.face_train_iterator = iter(cycle(self.face_train_loader))
        self.face_val_iterator = iter(cycle(self.face_val_loader))
        model_name = 'model.{}'.format(self.opt['model_name'])

        self.pase_cfg_path = config['pase_cfg_path']
        self.model = SVHFNet(self.pase_cfg_path).to(self.device)

        if config['load_model']:
            print('Load pretrained model: {}, {}'.format(config['res_ckpt_path'], config['pase_ckpt_path']))
            check_point = torch.load(config['res_ckpt_path'])  # cuda:1
            state_dict = check_point['model_state_dict']
            self.model.vis_stream.load_state_dict(state_dict)
            self.model.vis_stream.to(self.device)

            self.model.pase.load_pretrained(config['pase_ckpt_path'], load_last=True, verbose=True)
            self.model.aud_stream = AudioStream(self.model.pase)
            self.model.aud_stream.to(self.device)

        if config['multi_gpu']:
            print('Use Multi GPU.')
            # self.model = nn.DataParallel(self.model, device_ids=config['gpu_ids'])

        self.criterion = nn.CrossEntropyLoss()
        ### triplet loss setting
        self.distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)

        self.loss_func = losses.TripletMarginLoss(margin=0.2, reducer=reducer, distance=self.distance)
        self.mining_func = TripletMarginMiner(margin=0.2, type_of_triplets="semihard", distance=self.distance)

        if config['optim'] == 'SGD':
            print('Use SGD optimizer.')
            self.optim = torch.optim.SGD(params=self.model.parameters(), lr=config['lr'],
                                         momentum=0.9,
                                         weight_decay=0.0005)
        elif config['optim'] == 'Adam':
            print('Use Adam optimizer.')
            self.aud_optim = torch.optim.Adam(params=self.model.aud_stream.parameters(), lr=0.0001)
            self.vis_optim = torch.optim.Adam(params=self.model.vis_stream.parameters(), lr=0.001)


        self.aud_scheduler = torch.optim.lr_scheduler.StepLR(self.aud_optim, step_size=50,
                                                             gamma=0.9, last_epoch=-1)
        self.vis_scheduler = torch.optim.lr_scheduler.StepLR(self.vis_optim, step_size=50,
                                                             gamma=0.9, last_epoch=-1)

        self.savers = []
        self.timestamp = time.strftime("%Y-%h-%d:%H:%M")
        self.save_path = os.path.join(self.opt['save_path'], self.timestamp)
        self.savers.append(Saver(self.model.aud_stream, self.save_path,
                           max_ckpts=self.opt['max_ckpts'],
                           optimizer=self.aud_optim, prefix='pase-'))

        self.savers.append(Saver(self.model.vis_stream, self.save_path,
                           max_ckpts=self.opt['max_ckpts'],
                           optimizer=self.vis_optim, prefix='resnet-'))


        # -----------------------log and print ------------------------------ #
        self.tensorboard = self.opt['tensorboard']
        self.log = self.opt['log']
        if not os.path.exists(self.save_path) and self.log:
            os.makedirs(self.save_path)

        if self.tensorboard:
            self.writer = SummaryWriter(self.save_path)
        if self.log:
            self.logger = print_log(self.save_path, 'JVF-net')
            self.logger.write("Use tenoserboard: {}".format(self.tensorboard))
            self.logger.write(str(self.opt))

        self.epoch_time = AverageMeter()
        self.batch_time = AverageMeter()
        self.loss = AverageMeter()
        self.top1 = AverageMeter()

        self.last_iteration = 0
        self.bpe = len(self.voice_train_loader)
        self.print_freq = 1
        self.start_epoch = 1
        self.max_epoch = self.opt['epoch']


    def train_(self):
        self.logger.write('Start training..')
        self.logger.write('Batches per epoch: {}'.format(self.bpe))
        # self.aud_optim.zero_grad()
        # self.vis_optim.zero_grad()
        for epoch in range(self.max_epoch):
            self.model.train()
            epoch_end = time.time()
            for bidx in range(1, self.bpe+1):
                voice,  voice_emotion_label = next(self.voice_train_iterator)
                face, face_emotion_label = next(self.face_train_iterator)
                voice, face = voice.to(self.device), face.to(self.device)
                voice_emotion_label, face_emotion_label = voice_emotion_label.to(self.device), face_emotion_label.to(self.device)
                embeddings = self.model(face, voice)

                indices_tuple = self.mining_func(embeddings=embeddings['voice_emb'], labels=voice_emotion_label,
                                                 ref_emb=embeddings['face_emb'], ref_labels=face_emotion_label )
                indices_tuple = list(indices_tuple)
                concat_embeds = torch.cat((embeddings['voice_emb'], embeddings['face_emb']), dim=0)
                concat_labels = torch.cat((face_emotion_label,voice_emotion_label))
                indices_tuple[1] = indices_tuple[1] + embeddings['voice_emb'].shape[0]
                indices_tuple[2] = indices_tuple[2] + embeddings['voice_emb'].shape[0]
                loss = self.loss_func(concat_embeds, concat_labels, tuple(indices_tuple))

                self.aud_optim.zero_grad()
                self.vis_optim.zero_grad()
                loss.backward()
                self.aud_optim.step()
                self.vis_optim.step()
                self.loss.update(loss.item(),1)

                # ------------迭次信息输出--------------#
                if bidx % 10 == 0:
                    step = epoch * self.bpe + bidx
                    self.logger.write('Batch {}/{} (Epoch {}) step: {} Loss: {:f}:'. \
                                       format(bidx, self.bpe, epoch, step, self.loss.val))
            # -------------------轮次信息输出--------------------- #
            self.aud_scheduler.step()
            self.vis_scheduler.step()
            self.epoch_time.update(time.time() - epoch_end)

            self.logger.write('Train: epoch: [{0:}/{1:}]\t ' \
                              'Epoch Time: {2:.3f}, Loss_avg: {3:f},\t' \
                              'pase_lr: {4:f}, resnet_lr:{5:f}'.format(\
                               epoch, self.max_epoch,
                               self.epoch_time.val, self.loss.avg,
                               self.aud_optim.param_groups[0]['lr'], self.vis_optim.param_groups[0]['lr'],))

            pase_ckpt_file, resnet_ckpt_file = self.save(epoch)
            accuracies = self.validate(pase_ckpt_file, resnet_ckpt_file)

            if self.writer:
                self.writer.add_scalar('train/{}_loss'.format('total'), loss.item(), global_step=epoch)
                self.writer.add_scalar('train/{}_acc'.format('total'), accuracies, global_step=epoch)
            self.loss.reset()


    def validate(self, pase_ckpt_pth, res_ckpt_pth):
        self.logger.write("start validate")
        torch.cuda.empty_cache()

        res_ckpt = torch.load(res_ckpt_pth)  # cuda:1
        res_state_dict = res_ckpt['model_state_dict']
        self.model.vis_stream.load_state_dict(res_state_dict)
        self.model.vis_stream.to(self.device)

        # pase = wf_builder(self.pase_cfg_path).eval()     # read pre-trained model
        # self.aud_stream = AudioStream(pase)
        pase_ckpt = torch.load(pase_ckpt_pth)  # cuda:1
        pase_state_dict = pase_ckpt['model_state_dict']
        self.model.aud_stream.load_state_dict(pase_state_dict)
        self.model.aud_stream.to(self.device)

        accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1, avg_of_avgs=False)
        self.validate_VF = validate_for_VF_triplet(self.model.aud_stream, self.model.vis_stream, accuracy_calculator, batch_size=24)
        with torch.no_grad():
            accuracies = self.validate_VF.get_accuracy(self.face_train_data, self.voice_train_data)
            self.logger.write("Test set accuracy (Precision@1) = {} \n".format(accuracies["precision_at_1"]))

        return accuracies["precision_at_1"]


    def visualization(self, pase_ckpt_pth, res_ckpt_pth):
        # self.logger.write("start visualization")
        torch.cuda.empty_cache()

        # load face_network weight
        res_ckpt = torch.load(res_ckpt_pth)  # cuda:1
        res_state_dict = res_ckpt['model_state_dict']
        self.model.vis_stream.load_state_dict(res_state_dict)
        self.model.vis_stream.to(self.device)


        pase_ckpt = torch.load(pase_ckpt_pth)  # cuda:1
        pase_state_dict = pase_ckpt['model_state_dict']
        self.model.aud_stream.load_state_dict(pase_state_dict)
        self.model.aud_stream.to(self.device)

        self.inference = inference(self.model.aud_stream, self.model.vis_stream, self.distance)
        # self.inference.get_nearerst_images(self.face_train_data, self.voice_train_data)

        self.inference.get_pairs_images_from_voice(self.face_train_data, self.voice_train_data)

    def train_logger(self, preds, labels, losses, epoch, bidx, lrs):
        pass

    def save(self, epoch):
        self.pase_ckpt_file = os.path.join(self.save_path, '{}-checkpoint.pth'. \
                                           format('PASE'))
        torch.save({
            'epoch': epoch,
            'arch': self.model.aud_stream.__class__.__name__,  # class name
            'model_state_dict': self.model.aud_stream.state_dict(),
            'loss': self.loss,
        }, self.pase_ckpt_file)

        self.resnet_ckpt_file = os.path.join(self.save_path, '{}-checkpoint.pth'. \
                                             format('resnet'))
        torch.save({
            'epoch': epoch,
            'arch': self.model.vis_stream.__class__.__name__,  # class name
            'model_state_dict': self.model.vis_stream.state_dict(),
            'loss': self.loss,
        }, self.resnet_ckpt_file)


        if epoch % 50 == 0:
            resnet_best_file = os.path.join(self.save_path, '{}-epoch-{}.pth'. \
                                            format('resnet', epoch))
            shutil.copy(self.resnet_ckpt_file, resnet_best_file)
            pase_best_file = os.path.join(self.save_path, '{}-epoch-{}.pth'. \
                                          format('PASE', epoch))
            shutil.copy(self.pase_ckpt_file, pase_best_file)
        return self.pase_ckpt_file, self.resnet_ckpt_file

if __name__ == '__main__':

    with open('./option/SVHF.yaml', 'r') as config:
        config = yaml.load(config.read())
    Trainer = trainer(config)

    pase_ckpt_pth = '/home/fz/2-VF-feature/JVF-net/saved/JVF-net/2021-Jun-05:22:40/PASE-epoch-800.pth'
    res_ckpt_pth = '/home/fz/2-VF-feature/JVF-net/saved/JVF-net/2021-Jun-05:22:40/resnet-epoch-800.pth'
    Trainer.visualization(pase_ckpt_pth=pase_ckpt_pth, res_ckpt_pth=res_ckpt_pth)