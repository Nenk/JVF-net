import json
#import librosa
import argparse
import random
from random import shuffle
import numpy as np

import os




def main(opts):
    random.seed(opts.seed)
    spk2idx = {}

    data_cfg = {'train':{'data':[],
                         'speakers':[],
                         'emotions': []},
                'valid':{'data':[],
                         'speakers':[],
                         'emotions': []},
                'test':{'data':[],
                        'speakers':[]},
                'speakers':[],
                'emotions': []
                }
    with open(opts.train_scp, 'r') as train_f:
        train_files = train_f.readlines()[1:]
        # train_files = [l.rstrip() for l in train_f]
        shuffle(train_files)    # 随机排序
        if opts.valid_scp is None:
            N_valid_files = int(len(train_files) * opts.val_ratio)
            valid_files = train_files[:N_valid_files]
            train_files = train_files[N_valid_files:]

        print('Processing train file {:d}'.format(len(train_files)))
        data_cfg['train']['total_images'] = len(train_files)
        for ti, train_file in enumerate(train_files, start=1):
            actor_id, gender, vocal_channel, emotion, emotion_intensity, image_path = train_file.rstrip("\n").split(',')
            if actor_id not in data_cfg['speakers']:
                data_cfg['speakers'].append(actor_id)
            if emotion not in data_cfg['emotions']:
                data_cfg['emotions'].append(emotion)

            data_cfg['train']['data'].append({'filename': image_path,
                                              'speaker': actor_id,
                                              'emotion': emotion})

        # data_cfg['train']['total_images'] = train_dur
        # print('train images samples {:d}'.format(train_dur))

        # 从训练集中抽取的验证集
        print('Processing valid file {:d}'.format(len(valid_files))) # 可利用 end='\r' 换行输出
        data_cfg['valid']['total_images'] = len(valid_files)
        if opts.valid_scp is None:
            for ti, valid_file in enumerate(valid_files, start=1):
                actor_id, gender, vocal_channel, emotion, emotion_intensity, image_path= valid_file.rstrip("\n").split(',')
                if actor_id not in data_cfg['speakers']:
                    data_cfg['speakers'].append(actor_id)
                if emotion not in data_cfg['emotions']:
                    data_cfg['emotions'].append(emotion)
                # data_cfg['valid']['speakers'].append(actor_ID)
                data_cfg['valid']['data'].append({'filename':image_path,
                                                  'speaker':actor_id,
                                                  'emotion': emotion
                                                  })
            # data_cfg['valid']['total_images'] = valid_dur
            # print('test images samples {:d}'.format(valid_dur))

    # 创建专门的验证集
    if opts.valid_scp is not None:
        with open(opts.valid_scp, 'r') as valid_f:
            valid_files = [l.rstrip() for l in valid_f]
            valid_dur = 0
            for ti, valid_file in enumerate(valid_files, start=1):
                print('Processing valid file {:7d}/{:7d}'.format(ti, len(valid_files)), end='\r')
                spk = spk2idx[valid_file]
                if spk not in data_cfg['speakers']:
                    data_cfg['speakers'].append(spk)
                    data_cfg['valid']['speakers'].append(spk)
                data_cfg['valid']['data'].append({'filename':valid_file,
                                                  'spk':spk})

            # data_cfg['valid']['total_images'] = valid_dur
            # print()

    # 创建专门的测试集
    if opts.test_scp is not None:
        with open(opts.test_scp, 'r') as test_f:
            test_files = [l.rstrip() for l in test_f]
            test_dur = 0
            for ti, test_file in enumerate(test_files, start=1):
                print('Processing test file {:7d}/{:7d}'.format(ti, len(test_files)), end='\r')
                spk = spk2idx[test_file]
                if spk not in data_cfg['speakers']:
                    data_cfg['speakers'].append(spk)
                    data_cfg['test']['speakers'].append(spk)
                data_cfg['test']['data'].append({'filename':test_file,
                                                  'spk':spk})

            # data_cfg['test']['total_images'] = test_dur

    with open(opts.cfg_file, 'w') as cfg_f:
        cfg_f.write(json.dumps(data_cfg))  # python 对象转化为 json
        print(opts.cfg_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset/RAVDESS')   # 1
    parser.add_argument('--train_scp', type=str, default='dataset/RAVDESS/RAVDESS_image.csv')  # 2
    parser.add_argument('--valid_scp', type=str, default=None)
    parser.add_argument('--test_scp', type=str, default=None)  # 3
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio to take out of training'
                             'in utterances ratio (Def: 0.1).')
    parser.add_argument('--cfg_file', type=str, default='dataset/RAVDESS/RAVDESS_image_data.cfg')

    parser.add_argument('--seed', type=int, default=3)
    
    opts = parser.parse_args()
    main(opts)

