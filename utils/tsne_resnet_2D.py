# -*- coding: utf-8 -*-
import os
import time
import torch
import random
from model.SVHF import AudioStream, ResNet, SVHFNet
import seaborn as sns


sns.set_style('darkgrid')   # 黑色网格背景
sns.set_palette('hls')
# sns.color_palette("hls")
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from dataset import RAVDESS_voice_Dataset, RAVDESS_face_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def scatter(x, labels):
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("hls", num_classes))
    emotion_str = ['neutral',  'calm',  'happy',  'sad',  'angry', 'fearful', 'disgust',  'surprised']
    labels = np.array(labels)
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x_ = (x - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(aspect='equal')

    # for index, label in enumerate(labels):
    #
    #     ax.scatter(x_[index, 0], x_[index, 1], label=emotion_str[label],
    #                lw=0, s=16, color=palette[label])

    scatter = ax.scatter(x_[:,0], x_[:,1],
                         lw=0, s=16, color=palette[np.array(label)])

    # ax.legend( loc="upper right", title="name")
    ax.axis('off')
    ax.axis('tight')
    # ax = fig.gca(projection='3d')
    plt.xticks([])
    plt.yticks([])
    # plt.tight_layout()

    # add the labels for each digit corresponding to the label

    txts = []
    for i in range(0, num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x_[labels==i, :], axis=0)
        txt = ax.text(xtext-0.03, ytext, emotion_str[i], fontsize=16)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=3, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    fig.savefig('Tsne-RAVDESS-visual_net-valid.png', dpi = 600)
    plt.show()


def get_embeddings_from_model(data_root, data_cfg, model, model_ckpt_path):
    embeddings = np.array([])
    color = []
    label = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_train_data = RAVDESS_face_Dataset(data_root, data_cfg, split='valid')
    face_train_loader = DataLoader(face_train_data,
                                   batch_size=1,
                                   num_workers=8, shuffle=True, drop_last=True)
    if model_ckpt_path is not None:
        check_point = torch.load(model_ckpt_path)  # cuda:1
        state_dict = check_point['model_state_dict']
        model.load_state_dict(state_dict, strict = False)
    model.to(device)
    model.to().eval()
    iterator = iter(face_train_loader)
    for batch_idx in trange(1, len(face_train_loader)+1):

        imgs, target = next(iterator)
        imgs, target = imgs.to(device), target.to(device)
        embedding = model(imgs)
        embedding = embedding.detach().cpu().numpy().flatten()

        embeddings = np.concatenate((embeddings, embedding), axis=0)

        label.append(int(target)-1)
        color.append(plt.cm.hsv(int(target) / 8))
    embeddings = embeddings.reshape(len(color), -1)

    return embeddings, label, color


if __name__ == '__main__':
    face_data_root = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS/1 image-Actor1-24'
    face_data_cfg = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS/RAVDESS_image_data.cfg'

    # RAVDESS_image_pth = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS/RAVDESS_image.csv'
    model_pth = '/home/fz/2-VF-feature/JVF-net/saved/Resnet/2021-May-26:18:04/resnet-epoch-60.pth'
    vis_stream = ResNet()
    embeddings, label, color = get_embeddings_from_model(face_data_root, face_data_cfg, vis_stream, model_pth)

    print('embedding numbers: {:d}'.format(len(label)))
    tsne = TSNE(n_components=2).fit_transform(embeddings)
    scatter(tsne, label)
    print('tsne shape: {:}'.format(tsne.shape))




