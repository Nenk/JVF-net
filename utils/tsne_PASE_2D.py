# -*- coding: utf-8 -*-
import os
import time
import torch
import random
from model.SVHF import AudioStream, ResNet, SVHFNet
import seaborn as sns
from pase.models.frontend import wf_builder

sns.set_style('whitegrid')   # 黑色网格背景
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
import matplotlib.font_manager as font_manager
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def scatter(embeddings, labels, pase_ckpt_path):
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

    # sns.scatterplot(embeddings_[:, 0], embeddings_[:, 1], hue=labels, legend='full', palette=sns.color_palette("hls", num_classes))
    # scatter = ax.scatter(x, y, lw=0, s=16, c=palette[labels])
    font = font_manager.FontProperties(family='Times New Roman',
                                       weight='bold',
                                       style='normal', size=16)
    ax.legend(prop=font)
    legend = ax.legend(bbox_to_anchor=(0.90,1.1),loc="upper left")

    ax.axis('off')
    # ax.axis('tight')
    plt.xticks([])
    plt.yticks([])
    # plt.xlim([0,1])
    # plt.ylim([0,1])
    # plt.xticks(fontsize=14, family = "Times New Roman")
    # plt.yticks(fontsize=14, family = "Times New Roman")
    plt.tight_layout(rect=[0,0,1,1])

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

    fig.savefig('Tsne-RAVDESS-audio_net-train-JF-net-{}.png'.format(pase_ckpt_path.split('/')[-2]), dpi = 600)
    plt.show()


def get_embeddings_from_model(data_root, data_cfg, model):
    embeddings = np.array([])
    color = []
    label = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    voice_train_data = RAVDESS_voice_Dataset(data_root, data_cfg, split='train')
    voice_train_loader = DataLoader(voice_train_data,
                                   batch_size=1,
                                   num_workers=8, shuffle=True, drop_last=True)

    model.to(device)
    model.to().eval()
    iterator = iter(voice_train_loader)
    for batch_idx in trange(1, len(voice_train_loader)+1):

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
    voice_data_root = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS/2 wave-Actor1-24-16k'
    voice_data_cfg = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS/RAVDESS_voice_data.cfg'
    pase_cfg_path = '/home/fz/2-VF-feature/PASE/cfg/frontend/PASE.cfg'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # RAVDESS_image_pth = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS/RAVDESS_image.csv'

    # pase_ckpt_path = '/home/fz/2-VF-feature/PASE/saved_model/RAVDESS_ckpt/2021-May-22:11:00/FE_e+800.ckpt'
    # pase_cfg_path = '/home/fz/2-VF-feature/PASE/cfg/frontend/PASE.cfg'
    # pase = wf_builder(pase_cfg_path).eval()  # read pre-trained model
    # pase.load_pretrained(pase_ckpt_path, load_last=True, verbose=True)
    # vis_stream = AudioStream(pase)
    # res_ckpt_path = '/home/fz/2-VF-feature/JVF-net/saved/Resnet/2021-May-26:18:04/resnet-epoch-60.pth'
    # pase_ckpt_path_ori = '/home/fz/2-VF-feature/PASE/saved_model/RAVDESS_ckpt/2021-May-22:11:00/FE_e+800.ckpt'
    # model = SVHFNet(res_ckpt_path, pase_cfg_path, pase_ckpt_path_ori).to(device)

    pase = wf_builder(pase_cfg_path).eval()  # read pre-trained model
    aud_stream = AudioStream(pase)

    pase_ckpt_path ='/home/fz/2-VF-feature/JVF-net/saved/JVF-net/2021-Jun-05:22:40/PASE-epoch-800.pth'
    pase_ckpt = torch.load(pase_ckpt_path)  # cuda:1
    pase_state_dict = pase_ckpt['model_state_dict']
    aud_stream.load_state_dict(pase_state_dict)

    embeddings, label, color = get_embeddings_from_model(voice_data_root, voice_data_cfg, aud_stream)
    print('embedding numbers: {:d}'.format(len(label)))
    tsne = TSNE(n_components=2).fit_transform(embeddings)
    scatter(tsne, label, pase_ckpt_path)
    print('tsne shape: {:}'.format(tsne.shape))




