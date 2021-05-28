import os
import csv
import copy
import numpy as np
from random import shuffle
import shutil
import string


def get_dataset_files(data_dir, data_ext, celeb_ids):
    """
    从文件夹中读取voice或face数据
    """
    data_list = []
    # # rename image folder(face) name
    # for root, dirs, _ in sorted(os.walk(data_dir)):
    #     for dir in dirs:
    #         folder_pth = os.path.join(root, dir)
    #         folder = folder_pth[len(data_dir):].split('/')[1]
    #         celeb_name = celeb_ids.get(folder, folder)    # default_value不设置的话默认为None，设置的话即如果找不到则返回default设定的值
    #         if celeb_name != folder:
    #             new_folder_pth = os.path.join(root, celeb_name)
    #             os.rename(folder_pth, new_folder_pth)

    # read data directory
    for root, dirs, filenames in sorted(os.walk(data_dir)):
        for filename in filenames:
            if filename.endswith(data_ext):
                filepath = os.path.join(root, filename)
                folder = filepath[len(data_dir):].split('/')[1]
                celeb_name = celeb_ids.get(folder, folder) #  default_value不设置的话默认为None，设置的话即如果找不到则返回default设定的值
                if celeb_name != folder:
                    data_list.append({'folder_name':folder, 'name': celeb_name, 'filepath': filepath})

    return data_list


def get_voclexb_labels(voice_list, face_list, celeb_ids):
    """
    合并voice和face中的同类项目
    :param voice_list:
    :param face_list:
    :return:
    """
    voice_names = {item['name'] for item in voice_list}
    face_names = {item['name'] for item in face_list}
    names = voice_names & face_names
    voice_face_list = []
    label_dict = {}

    #  通过列表推导式 保留同类项
    voice_list = [item for item in voice_list if item['name'] in names]
    face_list = [item for item in face_list if item['name'] in names]

    names = list(sorted(names))      # 增加排序, 固定名字与序列号
    for step, item in enumerate(names):
        label_dict[item] = step
    # label_dict = dict(zip(names, range(len(names))))
    temp1 = []
    temp2 = []

    # 建立face-list,
    for item in face_list:
        item['id'] = label_dict[item['name']]

    # 建立联合组voice+face-list, 利用name来分类
    for name in names[:]:
        for item in voice_list:   # 为list增加序号label_id
            if name == item['name']:
                temp1.append(item['filepath'])
        for item in face_list:
            if name == item['name']:
                temp2.append(item['filepath'])
        voice_face_list.append({'name': name, 'id_num': label_dict[name], 'id': celeb_ids[name],
                               'voice_path': copy.deepcopy(temp1), 'face_path': copy.deepcopy(temp2)})
        print(name)
        temp1.clear()
        temp2.clear()

    return voice_face_list, face_list


def get_voclexb_csv(csv_files, voice_folder, face_folder):
    """
    从list.csv中读取路径, 写入list中,
    :param data_params:
    :return: 数据路径以及标签,speaker数量
    """
    csv_headers = ['name','id_num', 'id' ,'voice_path', 'face_path']
    face_csv_headers = ['folder_name', 'filepath', 'id', 'name']
    triplet_list = []
    actor_dict, actor_dict1 = {}, {}

    with open(csv_files) as f:
        lines = f.readlines()[1:]
        for line in lines:
            actor_ID, name, gender, nation, _ = line.rstrip("\n").split('\t')
            actor_dict[actor_ID] = name
            actor_dict1[name] = actor_ID

    face_list = get_dataset_files(face_folder, 'jpg', actor_dict)
    voice_list = get_dataset_files(voice_folder, 'wav', actor_dict)
    voice_face_list, face_list = get_voclexb_labels(voice_list, face_list, actor_dict1)
    # np.save('./dataset/voclexb-VGG_face-datasets/face_list.npy', face_list)
    np.save('./dataset/voclexb-VGG_face-datasets/voice_face_list.npy', voice_face_list)
    csv_pth = os.path.join('./dataset/voclexb-VGG_face-datasets/', 'voice_face_list.csv')
    print(csv_pth)
    with open(csv_pth,'w',newline='', ) as f:
        f_scv = csv.DictWriter(f, csv_headers, delimiter = ',', lineterminator = '\n')
        f_scv.writeheader()
        f_scv.writerows(voice_face_list)

    return len(actor_dict)

def get_RAVDESS_voice_csv(data_pth, csv_pth, data_ext):
    """
    从音频特征或图像文件夹中读取对应文件, 在csv中写入该文件路径,情感,身份,性别标签
    :param image_data_pth: 图像文件抽取中间为代表图像
    :param csv_pth: csv文件输出位置
    :param data_ext: .npy或者.png格式
    :return:
    """
    data_list = []
    list_name ={"image":"png", "voice":"wav", "mfcc":"mfcc", "fbank":"fbank", "spectrogram":"spectrogram"}

    file_ext = list_name[data_ext]
    headers = ['actor_id', 'gender', 'vocal_channel', 'emotion', 'emotion_intensity', 'voice_path']
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # read data directory
    for root, folders, filenames in os.walk(data_pth):      # 音频数据集根目录, 子目录, 文件名
        folders.sort()
        filenames.sort()
        for filename in filenames:
            if filename.endswith(file_ext):              # 校验文件后缀名, wav或者npy
                voice_path = os.path.join(root, filename)
                flag = filename.split('.')[0].split('-')
                if flag[0] == '01':  # only use video
                    gend = "female" if int(flag[6])%2 else "male"
                    data_list.append({'actor_id':flag[6], 'gender':gend, 'vocal_channel':flag[1],
                                      'emotion':flag[2], 'emotion_intensity':flag[3], 'voice_path': voice_path})
                    print("voice_{0:}_path:{1:}, actor:{2:}".format(data_ext, voice_path, flag[6]))

    print("sample numbers:{}".format(len(data_list)))

    csv_pth = os.path.join(csv_pth, 'RAVDESS_{}.csv'.format(file_ext))
    print("csv_pth:{}".format(csv_pth))
    with open(csv_pth,'w',newline='') as f:
        f_scv = csv.DictWriter(f,headers)
        f_scv.writeheader()
        f_scv.writerows(data_list)

def get_RAVDESS_face_csv(data_pth, csv_pth, data_ext):
    """
    从音频特征或图像文件夹中读取对应文件, 在csv中写入该文件路径,情感,身份,性别标签
    :param image_data_pth: 图像文件抽取中间为代表图像
    :param csv_pth: csv文件输出位置
    :param data_ext: .npy或者.png格式
    :return:
    """
    data_list = []
    list_name ={"image":"png", "voice":"wav", "mfcc":"mfcc", "fbank":"fbank", "spectrogram":"spectrogram"}

    file_ext = list_name[data_ext]
    headers = ['actor_id','gender','vocal_channel','emotion','emotion_intensity','image_path']
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # read data directory
    for root, folders, filenames in os.walk(data_pth):      # 音频数据集根目录, 子目录, 文件名
        folders.sort()
        filenames.sort()
        for filename in filenames:
            if filename.endswith(file_ext):              # 校验文件后缀名, wav或者npy
                image_path = os.path.join(root, filename)
                flag = root.split('/')[-1].split('-')
                if flag[0] == '01':  # only use video
                    gend = "female" if int(flag[6])%2 else "male"
                    data_list.append({'actor_id':flag[6], 'gender':gend, 'vocal_channel':flag[1],
                                      'emotion':flag[2], 'emotion_intensity':flag[3], 'image_path': image_path})
                    print("face_{0:}_path:{1:}, actor:{2:}".format(data_ext, image_path, flag[6]))

    print("sample numbers:{}".format(len(data_list)))

    csv_pth = os.path.join(csv_pth, 'RAVDESS_image.csv')
    print("csv_pth:{}".format(csv_pth))
    with open(csv_pth,'w',newline='') as f:
        f_scv = csv.DictWriter(f,headers)
        f_scv.writeheader()
        f_scv.writerows(data_list)

def csv_to_list(csv_files, val_ratio=0.1):
    """
    从list.csv中读取路径, 写入list中,
    :param data_params:
    :return: 数据路径以及标签,speaker数量
    """
    train_list = []
    test_list = []
    actor_num = []
    emotion_num = []

    with open(csv_files) as train_f:
        print("Read csv_files from: {}".format(csv_files))
        files = train_f.readlines()[1:]
        shuffle(files)
        N_valid_files = int(len(files[:]) * val_ratio)
        valid_files = files[:N_valid_files]
        train_files = files[N_valid_files:]

        for line in train_files[:]:
            actor_id, gender, vocal_channel, emotion, _, image_path = line.rstrip("\n").split(',')
            actor_num.append(int(actor_id))
            emotion_num.append(int(emotion))
            # voice_list.append({'filepath': wave_path, 'name_id': actor_ID, 'emotion_id': emotion})
            train_list.append({'image_path': image_path, 'actor_id': actor_id, 'emotion': emotion})

        for line in valid_files[:]:
            actor_id, gender, vocal_channel, emotion, _, image_path = line.rstrip("\n").split(',')
            actor_num.append(int(actor_id))
            emotion_num.append(int(emotion))
            # voice_list.append({'filepath': wave_path, 'name_id': actor_ID, 'emotion_id': emotion})
            test_list.append({'image_path': image_path, 'actor_id': actor_id, 'emotion': emotion})

    return train_list, test_list, max(actor_num), max(emotion_num)


def RAVDESS_csv_to_list():
    pass


if __name__ == '__main__':
    # get_RAVDESS_dataset(DATASET_PARAMETERS)
    # data_dir = 'data/RAVDESS/fbank'

    # csv_files = './dataset/voclexb-VGG_face-datasets/vox1_meta.csv'
    # voice_folder = '/home/fz/2-VF-feature/JVF-net/dataset/voclexb-VGG_face-datasets/2-voice-wav'
    # face_folder = '/home/fz/2-VF-feature/JVF-net/dataset/voclexb-VGG_face-datasets/1-face'
    # num = get_voclexb_csv(csv_files, voice_folder, face_folder)

    voice_data_pth = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS/2 wave-Actor1-24-16k'
    image_data_pth = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS/1 image-Actor1-24'
    csv_pth = "/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS"
    # get_RAVDESS_face_csv( image_data_pth, csv_pth, 'image')
    get_RAVDESS_voice_csv( voice_data_pth, csv_pth, 'voice')

