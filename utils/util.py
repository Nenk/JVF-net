import pandas as pd
import csv
import os
import sys
import torch
import shutil
import pickle
import logging
import json

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


class Saver(object):

    def __init__(self, model, save_path, max_ckpts=None, optimizer=None, prefix=''):
        self.model = model
        self.save_path = save_path
        self.ckpt_path = os.path.join(save_path, '{}-checkpoints'.format(prefix))
        self.max_ckpts = max_ckpts
        self.optimizer = optimizer
        self.prefix = prefix

    def save(self, model_name, step, loss, best_val=False):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ckpt_path = self.ckpt_path
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as ckpt_f:
                # read latest checkpoints
                ckpts = json.load(ckpt_f)
        else:
            ckpts = {'latest': [], 'current': []}

        model_path = '{}-epoch-{}.ckpt'.format(model_name, step)
        if best_val:
            model_path = 'best_' + model_path
        model_path = '{}{}'.format(self.prefix, model_path)

        # get rid of oldest ckpt, with is the frst one in list
        latest = ckpts['latest']
        if len(latest) > 0:
            todel = latest[0]
            if self.max_ckpts is not None:
                if len(latest) >= self.max_ckpts:
                    try:
                        print('Removing old ckpt {}'.format(os.path.join(save_path, todel)))
                        os.remove(os.path.join(save_path, todel))
                        latest = latest[1:]
                    except FileNotFoundError:
                        print('ERROR: ckpt is not there?')

        latest += [model_path]

        ckpts['latest'] = latest
        ckpts['current'] = model_path

        with open(ckpt_path, 'w') as ckpt_f:
            ckpt_f.write(json.dumps(ckpts, indent=2))

        st_dict = {'epoch': step,
                   'arch': self.model.__class__.__name__,  # class name
                   'model_state_dict': self.model.state_dict(),
                   'loss': loss
                   }

        if self.optimizer is not None:
            st_dict['optimizer'] = self.optimizer.state_dict()
        # now actually save the model and its weights
        # torch.save(self.model, os.path.join(save_path, model_path))
        torch.save(st_dict, os.path.join(save_path, model_path))

    def read_latest_checkpoint(self):
        ckpt_path = self.ckpt_path
        print('Reading latest checkpoint from {}...'.format(ckpt_path))
        if not os.path.exists(ckpt_path):
            print('[!] No checkpoint found in {}'.format(self.save_path))
            return None
        else:
            with open(ckpt_path, 'r') as ckpt_f:
                ckpts = json.load(ckpt_f)
            curr_ckpt = ckpts['current']
            return curr_ckpt

    def load_weights(self):
        save_path = self.save_path
        curr_ckpt = self.read_latest_checkpoint()
        if curr_ckpt is None:
            print('[!] No weights to be loaded')
            return False
        else:
            st_dict = torch.load(os.path.join(save_path,
                                              'weights_' + \
                                              curr_ckpt))
            if 'state_dict' in st_dict:
                # new saving mode
                model_state = st_dict['state_dict']
                self.model.load_state_dict(model_state)
                if self.optimizer is not None and 'optimizer' in st_dict:
                    self.optimizer.load_state_dict(st_dict['optimizer'])
            else:
                # legacy mode, only model was saved
                self.model.load_state_dict(st_dict)
            print('[*] Loaded weights')
            return True

    def load_ckpt_step(self, curr_ckpt):
        ckpt = torch.load(os.path.join(self.save_path,
                                       'weights_' + \
                                       curr_ckpt),
                          map_location='cpu')
        step = ckpt['step']
        return step

    def load_pretrained_ckpt(self, ckpt_file, load_last=False, load_opt=True,
                             verbose=True):
        model_dict = self.model.state_dict()
        st_dict = torch.load(ckpt_file,
                             map_location=lambda storage, loc: storage)
        if 'state_dict' in st_dict:
            pt_dict = st_dict['state_dict']
        else:
            # legacy mode
            pt_dict = st_dict
        all_pt_keys = list(pt_dict.keys())
        if not load_last:
            # Get rid of last layer params (fc output in D)
            allowed_keys = all_pt_keys[:-2]
        else:
            allowed_keys = all_pt_keys[:]
        # Filter unnecessary keys from loaded ones and those not existing
        pt_dict = {k: v for k, v in pt_dict.items() if k in model_dict and \
                   k in allowed_keys and v.size() == model_dict[k].size()}
        if verbose:
            print('Current Model keys: ', len(list(model_dict.keys())))
            print('Current Pt keys: ', len(list(pt_dict.keys())))
            print('Loading matching keys: ', list(pt_dict.keys()))
        if len(pt_dict.keys()) != len(model_dict.keys()):
            raise ValueError('WARNING: LOADING DIFFERENT NUM OF KEYS')
            print('WARNING: LOADING DIFFERENT NUM OF KEYS')
        # overwrite entries in existing dict
        model_dict.update(pt_dict)
        # load the new state dict
        self.model.load_state_dict(model_dict)
        for k in model_dict.keys():
            if k not in allowed_keys:
                print('WARNING: {} weights not loaded from pt ckpt'.format(k))
        if self.optimizer is not None and 'optimizer' in st_dict and load_opt:
            self.optimizer.load_state_dict(st_dict['optimizer'])


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


def cycle(dataloader):
    while True:
        for data, emotion_label in dataloader:

            yield data,  emotion_label


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
