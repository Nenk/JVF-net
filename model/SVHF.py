import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from torchsummary import summary
from submodule.resblock import Block, OptimizedBlock
# pase_path = os.path.abspath('/home/fz/2-VF-feature/pase')
# sys.path.append(pase_path)
# print('Add pase to system path:', pase_path)
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

# we pretrained the network by triplet loss
class ResNet(nn.Module):
    def __init__(self, ch=64, class_num=1000, activation=F.relu, include_top =False):
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

        # h = F.relu(h)
        # h = torch.sum(h, (2, 3))  # Global sum pooling.

        h = self.avgpool(h)
        # if not self.include_top:
        #     return x

        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        # h = self.fc2(h)

        return h

# RestNet for visual stream, PASE for audio stream / All network is pretrained.
class AudioStream(nn.Module):
    def __init__(self, pase):
        super().__init__()
        self.pase = pase  # (B, 100, 300) for 3s audio
        self.fc1 = nn.Linear(100*256, 128)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(2048, 128)
        # self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.pase(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # x = self.relu2(self.fc2(x))
        return x


class SVHFNet(nn.Module):
    def __init__(self, res_ckpt_path, pase_cfg_path, pase_ckpt_path):
        super().__init__()
        # m3 = model3.SVHFNet()
        self.vis_stream = ResNet()
        map_location = None if torch.cuda.is_available() else 'cpu'
        check_point = torch.load(res_ckpt_path, map_location=map_location)  # cuda:1
        state_dict = check_point['model_state_dict']
        self.vis_stream.load_state_dict(state_dict)

        pase = wf_builder(pase_cfg_path).eval()             # read pre-trained model
        pase.load_pretrained(pase_ckpt_path, load_last=True, verbose=True)
        self.aud_stream = AudioStream(pase)

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
    # pase_cfg_path = '../../PASE/cfg/PASE.cfg'
    # pase_ckpt_path = '../../PASE/model/FE_e199.ckpt'
    # res_ckpt_path = '../../saved/model3_bn/model_16.pt'
    net = ResNet()
    print(net)
    device = torch.device('cuda')
    net = net.to(device)
    summary(net, (3,224,224))

    # face_a = torch.empty((2, 3, 224, 224))
    # face_b = torch.empty((2, 3, 224, 224))
    # audio_a = torch.empty((2, 1, 48000))

    # net = SVHFNet(res_ckpt_path, pase_cfg_path, pase_ckpt_path)
    # output = net(face_a, face_b, audio)
    # print(output.shape)