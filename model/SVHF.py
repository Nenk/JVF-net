import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

import torch.onnx
import onnx
from onnx import shape_inference
import netron

from torchsummary import summary
from submodule.resblock import Block, OptimizedBlock

sys.path.append("/home/fz/2-VF-feature/JVF-net/model")
from pase.models.frontend import wf_builder
import model3


def weight_init(m, config):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        if config['weight_init'] == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight)
        elif config['weight_init'] == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif config['weight_init'] == 'gaussian':
            nn.init.normal_(m.weight, mean=0, std=0.01)
        else:
            pass

# we pretrained the network by triplet loss 11111
class ResNet(nn.Module):
    def __init__(self, ch=64, class_num=1000, activation=F.relu, include_top=False):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.activation = activation
        self.block1 = OptimizedBlock(3, ch)
        self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
        self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
        self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
        self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
        self.block6 = Block(ch * 16, ch * 16, activation=activation, downsample=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(1024, 128)
        # self.fc2 = nn.Linear(2048, 128)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)

        h = self.avgpool(h)
        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        # h = self.fc2(h)

        return h


class AudioStream(nn.Module):
    def __init__(self, pase):
        super().__init__()
        self.pase = pase  # (B, 100, 300) for 3s audio
        self.fc1 = nn.Linear(100*256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(2048, 128)
        # self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.pase(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn1(x)
        # x = self.relu2(self.fc2(x))
        return x


# RestNet for visual stream, PASE for audio stream / All network is pretrained.
class AudioStream_v2(nn.Module):
    def __init__(self, pase):
        super().__init__()
        self.pase = pase  # (B, 100, 300) for 3s audio
        self.fc1 = nn.Sequential(
                  nn.Linear(256, 1024),
                  nn.BatchNorm1d(1024),
                  nn.ReLU(inplace=False),
        )
        self.fc2 = nn.Sequential(
                  nn.ReLU(inplace=False),
                  nn.Linear(1024, 8),
                  # nn.BatchNorm1d(128),
        )
        # self.fc2 = nn.Linear(2048, 128)
        # self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.pase(x)
        x = F.avg_pool1d(x, kernel_size=x.shape[2], stride=1)  # 对所有C求平均值,并保证不同时长的音频特征有相同输出
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # if not self.include_top:
        #     return x
        x = self.fc2(x)
        return x


class SVHFNet(nn.Module):
    def __init__(self, pase_cfg_path):
        super().__init__()
        # self.vis_stream_ = ResNet()
        self.vis_stream = ResNet()
        # map_location = None if torch.cuda.is_available() else 'cpu'

        self.pase = wf_builder(pase_cfg_path).eval()   # read pre-trained model from pase
        self.aud_stream = AudioStream(self.pase)

        self.fc8 = nn.Linear(3072, 1024)
        self.bn8 = nn.BatchNorm1d(1024)
        self.relu8 = nn.ReLU()
        self.fc9 = nn.Linear(1024, 512)
        self.bn9 = nn.BatchNorm1d(512)
        self.relu9 = nn.ReLU()
        self.fc10 = nn.Linear(512, 2)

    def forward(self, face, audio):
        f_a_embedding_ = self.vis_stream(face)
        v_a_embedding = self.aud_stream(audio)

        # a_embedding = F.relu(a_embedding)
        pkg = {'face_emb': f_a_embedding_, 'voice_emb': v_a_embedding}

        # concat = torch.cat([f_a_embedding_, v_a_embedding], dim=1)
        #
        #
        # x = self.relu8(self.bn8(self.fc8(concat)))
        # x = self.relu9(self.bn9(self.fc9(x)))
        # x = self.fc10(x)
        return pkg

def get_network():
    pass


if __name__ == '__main__':

    device = torch.device('cuda')
    pase_cfg_path = '/home/fz/2-VF-feature/PASE/cfg/frontend/PASE.cfg'
    pase_ckpt_path =  '/home/fz/2-VF-feature/PASE/saved_model/RAVDESS_ckpt/2021-May-22:11:00/FE_e+800.ckpt'
    res_ckpt_path = '../../saved/model3_bn/model_16.pt'

    # load resnet
    net = ResNet()
    print(net)
    net = net.to(device)
    summary(net, (3, 224, 224))
    input = torch.rand(1, 3, 224, 224).to(device)
    #
    # # visualize for resnet
    # out = net(input)
    # onnx_path = "onnx_model_ResNet.onnx"
    # torch.onnx.export(net, input, onnx_path)
    # onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)
    # netron.start(onnx_path)

    # load PASE
    face_a = torch.empty((2, 3, 224, 224))
    input = torch.rand((2, 1, 16000)).to(device)   # batch_size * time * chunk_size

    pase = wf_builder(pase_cfg_path).eval()  # read pre-trained model from pase

    aud_stream = AudioStream(pase)
    aud_stream = aud_stream.to(device)
    print(aud_stream)
    summary(aud_stream, (1, 16000))

    # out = aud_stream(input)
    # onnx_path = "onnx_model_Pase.onnx"
    # torch.onnx.export(aud_stream, input, onnx_path)
    # onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)
    # netron.start(onnx_path)

    # output = model( audio_a.to(device))
    # print(output['voice_emb'].shape)