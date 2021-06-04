# -*- coding: utf-8 -*-
import os
import time
import torch
import random
from model.SVHF import AudioStream, ResNet, SVHFNet
import seaborn as sns


sns.set_style('whitegrid')   # 黑色网格背景
sns.set_palette('hls')

sns.set_context("paper", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from dataset import RAVDESS_voice_Dataset, RAVDESS_face_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def scatter(embeddings, labels):
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("hls", num_classes))
    emotion_str = np.array(['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'])
    labels = np.array(labels)
    embeddings_min, embeddings_max = np.min(embeddings, 0), np.max(embeddings, 0)
    embeddings_ = (embeddings - embeddings_min) / (embeddings_max - embeddings_min)
    x = embeddings_[:, 0]
    y = embeddings_[:, 1]

    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot(aspect='equal')

    for index, name in enumerate(['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']):
        indexs = np.where(labels == index)
        ax.scatter(x[indexs], y[indexs], c=palette[labels[indexs]], s=16, label=name, edgecolors='none')

    legend = ax.legend(bbox_to_anchor=(0.90,1.1),loc="upper left", fontsize=14, title_fontsize=14)
    plt.setp(legend.texts, family='Times New Roman')
    ax.axis('off')
    # ax.axis('tight')

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(rect=[0, 0, 1, 1])

    # add the labels for each digit corresponding to the label
    txts = []
    for i in range(0, num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(embeddings_[labels==i, :], axis=0)
        txt = ax.text(xtext-0.04, ytext, emotion_str[i], fontsize=16, family="Times New Roman")
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=3, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    fig.savefig('Tsne-RAVDESS-visual_net-valid-JF-net.png', dpi = 600)
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
    model_pth = '/home/fz/2-VF-feature/JVF-net/saved/JVF-net/2021-May-27:16:24/resnet-checkpoint-2021-May-27:16:24.pth'
    vis_stream = ResNet()

    embeddings, label, color = get_embeddings_from_model(face_data_root, face_data_cfg, vis_stream, model_pth)
    print('embedding numbers: {:d}'.format(len(label)))
    tsne = TSNE(n_components=2).fit_transform(embeddings)
    scatter(tsne, label)
    print('tsne shape: {:}'.format(tsne.shape))




