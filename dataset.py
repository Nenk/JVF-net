import os
import numpy as np
import torch
import librosa
import torchaudio
import random
import wave

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset, DataLoader


from PIL import Image
from scipy.io import wavfile


img_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def load_audio(audio_path, chunk_size=16000, fixed_offset=False, load_raw=True):
    y = librosa.load(audio_path, sr=None)
    y = y[0]
    if fixed_offset:
        offset = 0
    else:
        max_offset = y.shape[0] - chunk_size
        offset = random.randint(0, max_offset)
    y = y[offset:offset + chunk_size]
    if load_raw:
        y = np.expand_dims(y, axis=0)
        return y
    spect = librosa.Spectrogram(y)
    # for i in range(spect.shape[1]):
    #     f_bin = spect[:, i]
    #     f_bin_mean = np.mean(f_bin)
    #     f_bin_std = np.std(f_bin)
    #     spect[:, i] = (spect[:, i] - f_bin_mean) / (f_bin_std + 1e-7)
    # spect = np.expand_dims(spect, axis=0)
    return spect

def load_face(face_path):
    # NOTE: 3 channels are in BGR order
    img = Image.open(face_path).convert('RGB')
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()
    if img.size != (224, 224):    # 灰度图转为彩图
       img = img.resize((224, 224), resample=Image.BILINEAR)

    img = img_transform(img)
    return img


class VGG_Face_Dataset(Dataset):
    def __init__(self, face_list):
        # face_list = np.load(face_voice_dir, allow_pickle=True)
        self.face_list = face_list
        # self.speakers_num = len(self.face_list)  # 计算发言者数量

    def __getitem__(self, index):
        face_data = self.face_list[index]
        label = int(face_data['id'])
        real_face_path = face_data['image_path']
        real_face = load_face(real_face_path)

        return real_face, label

    def __len__(self):
        return len(self.face_list)


class RAVDESS_face_Dataset(Dataset):
    def __init__(self, data_root, data_cfg_file, split, verbose=True):

        self.data_root = data_root
        self.data_cfg_file = data_cfg_file
        self.label = []
        if not isinstance(data_cfg_file, str):
            raise ValueError('Please specify a path to a cfg '
                             'file for loading data.')

        with open(data_cfg_file, 'r') as data_cfg_f:
            self.data_cfg = json.load(data_cfg_f)
            self.spk_info = self.data_cfg['speakers']
            self.emo_info = self.data_cfg['emotions']
            if verbose:
                print('dataset.WavDataset:Found {} speakers info'.format(len(self.spk_info)))
                print('dataset.WavDataset:Found {} emotions info'.format(len(self.emo_info)))
                imgs = self.data_cfg[split]['data']
                print('dataset.WavDataset:Found {} files in {} split'.format(len(imgs), split))

                # spks = self.data_cfg[split]['speakers']
                # print('dataset.WavDataset:Found {} speakers in {} split'.format(len(spks), split))
                self.total_images = int(self.data_cfg[split]['total_images'])
            self.imgs = imgs
            for item in self.imgs:
                self.label.append(int(item['emotion']))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        uttname = self.imgs[index]['filename']
        actor_label = int(self.imgs[index]['speaker'])
        emotion_label = int(self.imgs[index]['emotion'])
        real_face = load_face(uttname)
        return real_face, emotion_label


class RAVDESS_voice_Dataset(Dataset):
    def __init__(self, data_root, data_cfg_file, split,
                 sr=16000,
                 cache_on_load=False,
                 verbose=True,
                 *args, **kwargs):
        # sr: sampling rate, (Def: None, the one in the wav header)
        self.sr = sr
        self.data_root = data_root
        self.cache_on_load = cache_on_load
        self.data_cfg_file = data_cfg_file
        if not isinstance(data_cfg_file, str):
            raise ValueError('Please specify a path to a cfg '
                             'file for loading data.')

        self.split = split
        self.return_spk = False
        self.label = []
        with open(data_cfg_file, 'r') as data_cfg_f:
            self.data_cfg = json.load(data_cfg_f)
            self.spk_info = self.data_cfg['speakers']
            self.emo_info = self.data_cfg['emotions']
            if verbose:
                print('dataset.WavDataset:Found {} speakers info'.format(len(self.spk_info)))
                print('dataset.WavDataset:Found {} emotions info'.format(len(self.emo_info)))
                wavs = self.data_cfg[split]['data']
                print('dataset.WavDataset:Found {} files in {} split'.format(len(wavs), split))

                # spks = self.data_cfg[split]['speakers']
                # print('dataset.WavDataset:Found {} speakers in {} split'.format(len(spks), split))
                self.total_wav_dur = int(self.data_cfg[split]['total_wav_dur'])
            self.wavs = wavs
            for item in self.wavs:
                self.label.append(int(item['emotion']))
    def __len__(self):
        return len(self.wavs)

    def retrieve_cache(self, fname, cache):
        # wav, rate = librosa.load(fname, sr=None)  # torchaudio

        wav, rate = torchaudio.load(fname)
        wav = wav.numpy().squeeze()
        #fix in case wav is stereo, in which case
        #pick first channel only
        if wav.ndim > 1:
            wav = wav[:,0]
        wav = wav.astype(np.float32)
        if self.cache_on_load:
            cache[fname] = wav, split='valid'
        return wav

    def __getitem__(self, index):
        uttname = self.wavs[index]['filename']
        wname = os.path.join(self.data_root, uttname)
        actor_label = int(self.wavs[index]['speaker'])
        emotion_label = int(self.wavs[index]['emotion'])
        wav = load_audio(wname, chunk_size=16000, fixed_offset=False, load_raw=True)
        # if len(rets) == 1:
        #     return rets[0]

        return wav, emotion_label


def custom_collate_fn(batch):
    real_audio = [torch.from_numpy(item[0]) for item in batch]
    face_a = [item[1] for item in batch]
    face_b = [item[2] for item in batch]
    gt = [item[3] for item in batch]
    real_audio = torch.stack(real_audio, dim=0)
    face_a = torch.stack(face_a, dim=0)
    face_b = torch.stack(face_b, dim=0)
    gt = torch.cat(gt, dim=0)
    return [real_audio, face_a, face_b, gt]


if __name__ == '__main__':
    data_root = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS'
    data_cfg = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS/RAVDESS_image_data.cfg'
    trans = None
    voice_dataset = RAVDESS_face_Dataset(data_root, data_cfg, split='valid')

    voice_loader = DataLoader(voice_dataset, batch_size=24, shuffle=True, drop_last=False, num_workers=0)
    voice_loader = iter(voice_loader)
    for i in range(10):

        data, label = next(voice_loader)
        print(data.shape)  # (B, 1, 512, 300)
        print(label.shape)  # (B)

    # for step, (data, label) in enumerate(voice_loader):
    #     print(data.shape)  # (B, 1, 512, 300)
    #     print(label.shape)  # (B)
    #     print(step)