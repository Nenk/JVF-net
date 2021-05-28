from __future__ import division

import torch
import numpy as np
import torchvision.transforms as transforms
from model.SVHF import ResNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import util
from dataset import VGG_Face_Dataset, load_face
from utils.parse_dataset import csv_to_list

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import accuracy_calculator

configure = {
    'network': dict(
        type='resnet',
        class_num = 24),

    'testing': dict(
        start_epoch=0,
        start_iteration = 0,
        batch_size = 32,
        max_epoch= 3000,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        gamma=0.9,  # "lr_policy: step"
        step_size=1000,  # "lr_policy: step"
        interval_validate=1000,
    ),

    'csv_list': '/home/fz/2-VF-feature/JVF-net/dataset/voclexb-VGG_face-datasets/face_list.csv',
    'log_dir': '../log/',
    'ckpt_dir':'../saved/',
}


def one_image_test(image_path, model, device):

    img = load_face(image_path)
    # img1 = np.expand_dims(img, axis=0)
    img1 = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        inputs = img1.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
    print("img : {}, predict as : {}".format(image_path, predicted[0]))


class YourCalculator(AccuracyCalculator):
    def calculate_precision_at_2(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 2)

    def calculate_fancy_mutual_info(self, query_labels, cluster_labels, **kwargs):
        return 1

    def requires_clustering(self):
        return super().requires_clustering() + ["fancy_mutual_info"]

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_2"]


class validate_for_triplet():
    def __init__(self, model, accuracy_calculator, batch_size):
        self.model = model
        self.accuracy_calculator = accuracy_calculator
        self.batch_size = batch_size

    def get_accuracy(self, train_set, test_set):
        train_embeddings, train_labels = self.get_all_embeddings(train_set, self.model)
        test_embeddings, test_labels = self.get_all_embeddings(test_set, self.model)
        accuracies = self.accuracy_calculator.get_accuracy(test_embeddings,
                                                      train_embeddings,
                                                      test_labels,
                                                      train_labels,
                                                      False)
        return accuracies


    def get_all_embeddings(self, dataset, model):
        tester = testers.BaseTester(batch_size=self.batch_size,
                                    dataloader_num_workers=8,)
        return tester.get_all_embeddings(dataset, model)


def validate_for_softmax(test_loader, model, device):
    correct = 0
    total = 0
    acorrect_list =[]
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
            correct +=  predicted.eq(targets).sum().item()

            print("batch_ix [{:4d}/{:4d}] test acc: {:6.3f}% ".
                  format(batch_index, len(test_loader), 100.0 * correct/total))

        print("average test acc: {:6.3f}% ".format( 100.0 * correct/total))

if __name__ == '__main__':

    train_cfg = configure['testing']
    net_cfg = configure['network']
    face_list, actor_num = csv_to_list(configure['csv_list'])
    ckpt_file_pth= configure['ckpt_dir'] + 'resnet-checkpoint-2021-04-05,14,31.pth'

    # load dataset
    face_dataset = VGG_Face_Dataset(face_list, 'train')
    face_loader = DataLoader(face_dataset, batch_size=train_cfg['batch_size'], drop_last=False,
                             shuffle=True, num_workers=0, pin_memory=True)
    face_image = '/home/fz/2-VF-feature/JVF-net/dataset/voclexb-VGG_face-datasets/1-face/id10258/00000002.jpg'

    # define netowrk
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if net_cfg['type'] == 'resnet':
        model = ResNet(class_num=actor_num, include_top=True)

    # load network
    checkpoint = torch.load(ckpt_file_pth)
    ckpt = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint['model_state_dict'], strict =True)

    # print the test information
    batch_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    top5 = util.AverageMeter()

    # one_image_prediction
    model.to(device)
    # one_image_test(image_path=face_image, model=model, device=device)
    # validate(test_loader=face_loader,model=model,device=device)

    # for batch_idx, (imgs, target, img_files, class_ids) in tqdm.tqdm(
    #         enumerate(self.val_loader), total=len(self.val_loader),
    #         desc='Valid iteration={} epoch={}'.format(self.iteration, self.epoch), ncols=80, leave=False):
    #
    #     gc.collect()
    #     if self.cuda:
    #         imgs, target = imgs.cuda(), target.cuda(async=True)
    #     imgs = Variable(imgs, volatile=True)
    #     target = Variable(target, volatile=True)
    #
    #     output = self.model(imgs)
    #     loss = self.criterion(output, target)
    #
    #     if np.isnan(float(loss.item())):
    #         raise ValueError('loss is nan while validating')
    #
    #     # measure accuracy and record loss
    #     prec1, prec5 = utils.accuracy(output.data, target.data, topk=(1, 5))
    #     losses.update(loss.item(), imgs.size(0))
    #     top1.update(prec1[0], imgs.size(0))
    #     top5.update(prec5[0], imgs.size(0))
    #
    #     # measure elapsed time
    #     batch_time.update(time.time() - end)
    #     end = time.time()
    #     if batch_idx % self.print_freq == 0:
    #         log_str = 'Test: [{0}/{1}/{top1.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
    #                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
    #                   'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
    #                   'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t' \
    #                   'Prec@5: {top5.val:.3f} ({top5.avg:.3f})\t'.format(
    #             batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
    #             batch_time=batch_time, loss=losses, top1=top1, top5=top5)
    #         print(log_str)
    #         self.print_log(log_str)
    # if self.cmd == 'test':
    #     log_str = 'Test: [{0}/{1}/{top1.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
    #               'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
    #               'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
    #               'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t' \
    #               'Prec@5: {top5.val:.3f} ({top5.avg:.3f})\t'.format(
    #         batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
    #         batch_time=batch_time, loss=losses, top1=top1, top5=top5)
    #     print(log_str)
    #     self.print_log(log_str)
    #
    # if self.cmd == 'train':
    #     is_best = top1.avg > self.best_top1
    #     self.best_top1 = max(top1.avg, self.best_top1)
    #     self.best_top5 = max(top5.avg, self.best_top5)
    #
    #     log_str = 'Test_summary: [{0}/{1}/{top1.count:}] epoch: {epoch:} iter: {iteration:}\t' \
    #               'BestPrec@1: {best_top1:.3f}\tBestPrec@5: {best_top5:.3f}\t' \
    #               'Time: {batch_time.avg:.3f}\tLoss: {loss.avg:.4f}\t' \
    #               'Prec@1: {top1.avg:.3f}\tPrec@5: {top5.avg:.3f}\t'.format(
    #         batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
    #         best_top1=self.best_top1, best_top5=self.best_top5,
    #         batch_time=batch_time, loss=losses, top1=top1, top5=top5)
    #     print(log_str)
    #     self.print_log(log_str)
    #
    #     checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pth.tar')
    #     torch.save({
    #         'epoch': self.epoch,
    #         'iteration': self.iteration,
    #         'arch': self.model.__class__.__name__,
    #         'optim_state_dict': self.optim.state_dict(),
    #         'model_state_dict': self.model.state_dict(),
    #         'best_top1': self.best_top1,
    #         'batch_time': batch_time,
    #         'losses': losses,
    #         'top1': top1,
    #         'top5': top5,
    #     }, checkpoint_file)
    #     if is_best:
    #         shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
    #     if (self.epoch + 1) % 10 == 0:  # save each 10 epoch
    #         shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'checkpoint-{}.pth.tar'.format(self.epoch)))
    #
    #     if training:
    #         self.model.train()



