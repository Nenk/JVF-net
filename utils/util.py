import pandas as pd
import csv
import os
import sys
import torch
import shutil
import pickle
import logging


class Logger(object):
    """
    日志模块记录 所输出的文本
    """
    def __init__(self, save_path, times):
        self.logger = logging.getLogger('lossesLogger')
        self.logFile = save_path
        if not os.path.exists(self.logFile):
            os.makedirs(self.logFile)
            # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        handler = logging.FileHandler(self.logFile + '/logFile_{0}.log'.format(times))
        # handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(hdlr=handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("starting logger model...")

    def write(self, out):
        console_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info(out)

class print_log(object):
    def __init__(self, log_dir, times):
        self.log_file = os.path.join(log_dir, 'logFile_{0}.log'.format(times))
        self.write("starting logger model...")

    def write(self, str):
        with open(self.log_file, 'a') as f:
            f.write(str + '\n')
            print(str)

def load_state_dict_VGG(model, weight):

    pretrained_dict = torch.load(weight)
    model_dict = model.state_dict()
    for k, v in pretrained_dict.items():
        if 'classifier' not in k:
            model_dict[k] = v

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    # model_dict.update(pretrained_dict)  # 相同键更新，新的键和键值添加
    model.load_state_dict(model_dict, strict=False)
    # model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root),strict=False)


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            if 'fc' in name:
                continue
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                print(name)
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def get_id_label_map(meta_file):
    N_IDENTITY = 9131  # total number of identities in VGG Face2
    N_IDENTITY_PRETRAIN = 8631  # the number of identities used in training by Caffe
    identity_list = meta_file
    df = pd.read_csv(identity_list, sep=',\s+', quoting=csv.QUOTE_ALL, encoding="utf-8")
    df["class"] = -1
    df.loc[df["Flag"] == 1, "class"] = range(N_IDENTITY_PRETRAIN)
    df.loc[df["Flag"] == 0, "class"] = range(N_IDENTITY_PRETRAIN, N_IDENTITY)
    # print(df)
    key = df["Class_ID"].values
    val = df["class"].values
    id_label_dict = dict(zip(key, val))
    return id_label_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    output_sorted, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def getLabel(attribution = "",file_path = ""):
    all_img = os.listdir(file_path)
    dic = []
    lable_number = []
    for img in all_img:
        filename = os.path.join(file_path,img)
        if attribution == "identity":
            label = img[:-4].split('-')[-1]
            lable_number.append(int(label))
        elif attribution == "emotion":
            label = img[:-4].split('-')[2]
            lable_number.append(int(label))
        dic.append({'img':filename,'lbl':int(label)-1})
    return dic, max(lable_number)

def getLabel_from_RAVDESS(attribution = "",file_path = ""):

    dic = []
    lable_number = []
    for root, dirs, filenames in os.walk(file_path):
        filenames.sort()
        flag = root.split('/')[-1].split('-')
        for img in filenames:
            filename = os.path.join(root,img)
            if attribution == "identity":
                label = flag[6]
                lable_number.append(int(label))
            elif attribution == "emotion":
                label = flag[2]
                lable_number.append(int(flag[2]))
            dic.append({'img':filename,'lbl':int(label)-1})
    return dic, max(lable_number)

if __name__ == '__main__':
    accuracy
