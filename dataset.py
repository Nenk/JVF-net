import os
import numpy as np
import torch
import librosa
import random
import wave

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
from scipy.io import wavfile

Transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def load_face(face_path):
    # NOTE: 3 channels are in BGR order
    img = Image.open(face_path).convert('RGB')
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()
    if img.size != (224, 224):    # 灰度图转为彩图
       img = img.resize((224, 224), resample=Image.BILINEAR)

    img = Transform(img)
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

class RAVDESS_Face_Dataset(Dataset):
    def __init__(self, face_list):
        self.label = []
        self.face_list = face_list
        for face_data in self.face_list:
            self.label.append(int(face_data['emotion']))



    def __getitem__(self, index):
        face_data = self.face_list[index]
        actor_label = int(face_data['actor_id'])
        emotion_label = int(face_data['emotion'])
        real_face_path = face_data['image_path']
        real_face = load_face(real_face_path)
        return real_face, emotion_label

    def __len__(self):
        return len(self.face_list)


class WavDataset(Dataset):

    def __init__(self, data_root, data_cfg_file, split,
                 transform=None, sr=None,
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
        self.transform = transform

        with open(data_cfg_file, 'r') as data_cfg_f:
            self.data_cfg = json.load(data_cfg_f)
            self.spk_info = self.data_cfg['speakers']
            if verbose:
                print('dataset.WavDataset:Found {} speakers info'.format(len(self.spk_info)))
                wavs = self.data_cfg[split]['data']
                print('dataset.WavDataset:Found {} files in {} split'.format(len(wavs), split))
                # spks = self.data_cfg[split]['speakers']
                # print('dataset.WavDataset:Found {} speakers in {} split'.format(len(spks), split))
                self.total_wav_dur = int(self.data_cfg[split]['total_wav_dur'])

            self.wavs = wavs
        self.wav_cache = {}


    def __len__(self):
        return len(self.wavs)

    def retrieve_cache(self, fname, cache):
        if (self.cache_on_load or self.preload_wav) and fname in cache:
            return cache[fname]
        else:
            wav, rate = librosa.load(fname)
            wav = wav.numpy().squeeze()
            #fix in case wav is stereo, in which case
            #pick first channel only
            if wav.ndim > 1:
                wav = wav[:,0]
            wav = wav.astype(np.float32)
            if self.cache_on_load:
                cache[fname] = wav
            return wav

    def __getitem__(self, index):
        if sample_probable(self.zero_speech_p):
            wav = zerospeech(int(5 * 16e3))
            if self.zero_speech_transform is not None:
                wav = self.zero_speech_transform(wav)
        else:
            uttname = self.wavs[index]['filename']
            wname = os.path.join(self.data_root, uttname)
            wav = self.retrieve_cache(wname, self.wav_cache)
            if self.transform is not None:
                wav = self.transform(wav)
        rets = [wav]
        if self.return_uttname:
            rets = rets + [uttname]
        if self.return_spk:
            rets = rets + [self.spk2idx[self.wavs[index]['speaker']]]
        if len(rets) == 1:
            return rets[0]
        else:
            return rets


class RAVDESS_voice_Dataset(Dataset):
    def __init__(self, data_root, data_cfg_file, split, fixed_offset, load_raw=False):
        # self.data_dir = data_dir
        self.fixed_offset = fixed_offset
        self.load_raw = load_raw





    def __getitem__(self, p_index):
        n_index = p_index

        positive = self.all_triplets[p_index]

        while(n_index == p_index):
            n_index = np.random.randint(0, self.speakers_num)   # 计算 0~1225之间的随机数

        negative = self.all_triplets[n_index]

        real_audio_path = positive['voice_path'][np.random.randint(0, len(positive['voice_path']))]
        real_face_path =  positive['face_path'][np.random.randint(0, len(positive['face_path']))]
        fake_face_path = negative['face_path'][np.random.randint(0, len(negative['face_path']))]

        real_audio = self.load_audio(real_audio_path)
        real_face = load_face(real_face_path)
        fake_face = load_face(fake_face_path)
        which_side = random.randint(0, 1)
        if which_side == 0:
            ground_truth = torch.LongTensor([0])
            face_a = real_face
            face_b = fake_face
        else:
            ground_truth = torch.LongTensor([1])
            face_a = fake_face
            face_b = real_face
        return real_audio, face_a, face_b, ground_truth

    def load_audio(self, audio_path):
        y = librosa.load(audio_path)
        y = y[0]
        if self.fixed_offset:
            offset = 0
        else:
            max_offset = y.shape[0] - 48000
            offset = random.randint(0, max_offset)
        y = y[offset:offset+48000]
        if self.load_raw:
            y = np.expand_dims(y, axis=0)
            return y
        spect = Face_Voice_Dataset.get_spectrogram(y)
        for i in range(spect.shape[1]):
            f_bin = spect[:, i]
            f_bin_mean = np.mean(f_bin)
            f_bin_std = np.std(f_bin)
            spect[:, i] = (spect[:, i] - f_bin_mean) / (f_bin_std + 1e-7)
        spect = np.expand_dims(spect, axis=0)
        return spect

    def __len__(self):
        return len(self.all_triplets)

    @staticmethod
    def get_spectrogram(y, n_fft=1024, hop_length=160, win_length=400, window='hamming'):
        y_hat = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        y_hat = y_hat[:-1, :-1]
        D = np.abs(y_hat)
        return D


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

    # dataset = Dataset( './dataset/voclexb-VGG_face-datasets/voice_face_list.npy', 'train', False)
    # loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True, num_workers=8, collate_fn=custom_collate_fn)
    #
    # for step, (real_audio, face_a, face_b, ground_truth) in enumerate(loader):
    #     print(real_audio.shape)  # (B, 1, 512, 300)
    #     print(face_a.shape)  # (B, 3, 224, 224)
    #     print(face_b.shape)
    #     print(ground_truth.shape)  # (B)

    face_dataset = VGG_Face_Dataset('./dataset/voclexb-VGG_face-datasets/voice_face_list.npy', 'train')
    face_loader = DataLoader(face_dataset, batch_size=24, shuffle=True, drop_last=False, num_workers=8)

    for step, (real_face, ground_truth) in enumerate(face_loader):
        print(real_face.shape)  # (B, 1, 512, 300)
        print(ground_truth.shape)  # (B)
        print(step)