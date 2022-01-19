import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
import torch
import os
import argparse

import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str,
                        default='/media/database/data4/wf/GraphNet-FER/work01/datasets/RAF-DB',
                        # default='E:/DataSet/RAF-DB',
                        help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--test_num', type=int, default=11)
    return parser.parse_args()


class RAFNeutralImage(data.Dataset):
    def __init__(self, raf_path, label_index, phase=None, transform=None):
        self.noise = {}
        self.raf_path = raf_path
        self.phase = phase
        self.transform = transform
        NAME_COLUMN = 0
        LABEL_COLUMN = 1

        df = pd.read_csv(os.path.join(self.raf_path, 'basic/EmoLabel/list_patition_label.txt'), sep=' ',
                         header=None)
        df = df[df[LABEL_COLUMN] == 7]

        if phase == 'train':
            df_train = df[df[NAME_COLUMN].str.startswith('train')]
            file_names = df_train.iloc[:, NAME_COLUMN].values
            self.label = df_train.iloc[:, LABEL_COLUMN].values - 1
            self.file_paths = []
            # use raf aligned images for training/testing
            for f in file_names:
                f = f.split(".")[0]
                f = f + "_aligned.jpg"
                path = os.path.join(self.raf_path, 'basic/Image/aligned', f)
                self.file_paths.append(path)
        elif phase == 'test':
            df_test = df[df[NAME_COLUMN].str.startswith('test')]
            file_names = df_test.iloc[:, NAME_COLUMN].values
            self.label = df_test.iloc[:, LABEL_COLUMN].values - 1
            self.file_paths = []
            # use raf aligned images for training/testing
            for f in file_names:
                f = f.split(".")[0]
                f = f + "_aligned.jpg"
                path = os.path.join(self.raf_path, 'basic/Image/aligned', f)
                self.file_paths.append(path)
        else:
            df_train = df[df[NAME_COLUMN].str.startswith('train')]
            file_names = df_train.iloc[:, NAME_COLUMN].values
            self.file_paths = []
            # use raf aligned images for training/testing
            for f in file_names:
                f = f.split(".")[0]
                f = f + "_aligned.jpg"
                path = os.path.join(self.raf_path, 'basic/Image/aligned', f)
                self.file_paths.append(path)

            df_test = df[df[NAME_COLUMN].str.startswith('test')]
            file_names = df_test.iloc[:, NAME_COLUMN].values
            # use raf aligned images for training/testing
            for f in file_names:
                f = f.split(".")[0]
                f = f + "_aligned.jpg"
                path = os.path.join(self.raf_path, 'basic/Image/aligned', f)
                self.file_paths.append(path)

        self.imageSize = len(self.file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # cv2.imwrite("img", image)
        # image = image[:, :, ::-1]  # BGR to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.copy()
        if self.transform is not None:
            image = self.transform(image)
        mot = self.noise.get(idx, None)
        if mot is None:
            mot = torch.randn(512)
            self.noise[idx] = mot

        return image, mot, idx


def getRAFdata():
    args = parse_args()

    # 加载train data
    # data_transforms = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    #     transforms.RandomErasing(scale=(0.02, 0.25))
    # ])
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, ], [0.229, ]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = RAFNeutralImage(args.raf_path, 0, phase='train', transform=data_transforms)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    test_dataset = RAFNeutralImage(args.raf_path, 0, phase='test', transform=data_transforms)
    print('Test set size:', test_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    all_dataset = RAFNeutralImage(args.raf_path, 0, transform=data_transforms)
    print('All set size:', all_dataset.__len__())
    all_loader = torch.utils.data.DataLoader(all_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              shuffle=True,
                                              pin_memory=True)
    return train_loader, test_loader, all_loader
    # return train_dataset, test_dataset

def Affectparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--affect_label_path', type=str,
                        default='/media/database/data4/wf/GraphNet-FER/work01/datasets/AffectNet/Manually_Annotated_file_lists',
                        # default='../../datasets/AffectNet/Manually_Annotated_file_lists',
                        help='AffectNet label path.')
    parser.add_argument('--affect_dataset_path', type=str,
                        default='/media/database/data3/Database/AffectNet/AffectNet',
                        help='AffectNet dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


class AffectNeutralImage(data.Dataset):
    def __init__(self, label_path, dataset_path, phase=None, transform=None):
        self.noise = {}
        self.phase = phase
        self.transform = transform
        self.label_path = label_path
        self.dataset_path = dataset_path
        self.nodes = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger']

        df1 = pd.read_csv(os.path.join(self.label_path, 'training.csv'),
                          header=None, low_memory=False).drop(0)
        df1['usage'] = ['training'] * df1.shape[0]
        df2 = pd.read_csv(os.path.join(self.label_path, 'validation.csv'),
                          header=None, low_memory=False).drop(0)
        df2['usage'] = ['validation'] * df2.shape[0]
        df = pd.concat([df1, df2], axis=0)
        df_name = df.iloc[:, 0]
        df_usage = df.iloc[:, -1]
        df_label = df.iloc[:, 6]
        df = pd.concat([df_name, df_label, df_usage], axis=1)
        new_col = [0, 1, 2]
        df.columns = new_col
        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype('int')
        df = df[df[LABEL_COLUMN] == 0]

        usage_column = 2
        df_train = df[df[usage_column] == 'training']
        df_val = df[df[usage_column] == 'validation']

        if phase == 'train':
            file_names = df_train.iloc[:, NAME_COLUMN].values
            self.label = df_train.iloc[:, LABEL_COLUMN].values
            self.file_paths = []
            for f in file_names:
                f = f.split(".")[0]
                f = f + ".jpg"
                path = os.path.join(self.dataset_path, f)
                self.file_paths.append(path)

        elif phase == 'test':
            file_names = df_val.iloc[:, NAME_COLUMN].values
            self.label = df_val.iloc[:, LABEL_COLUMN].values
            self.file_paths = []
            for f in file_names:
                f = f.split(".")[0]
                f = f + ".jpg"
                path = os.path.join(self.dataset_path, f)
                self.file_paths.append(path)

        else:
            file_names = df.iloc[:, NAME_COLUMN].values
            self.label = df.iloc[:, LABEL_COLUMN].values
            self.file_paths = []
            for f in file_names:
                f = f.split(".")[0]
                f = f + ".jpg"
                path = os.path.join(self.dataset_path, f)
                self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image, label, index = None, None, None
            return image, label, index
        else:
            image = image.copy()
            if self.transform is not None:
                image = self.transform(image)
            mot = self.noise.get(idx, None)
            if mot is None:
                mot = torch.randn(512)
                self.noise[idx] = mot

        return image, mot, idx


def getAffectdata():
    args = Affectparse_args()
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        transforms.Normalize([0.485, ], [0.229, ]),
        # transforms.RandomErasing(scale=(0.02, 0.25))
    ])
    train_dataset = AffectNeutralImage(args.affect_label_path, args.affect_dataset_path,
                                       phase='train', transform=data_transforms)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    test_dataset = AffectNeutralImage(args.affect_label_path, args.affect_dataset_path,
                                      phase='test', transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              shuffle=False,
                                              pin_memory=True)
    test_len = test_dataset.__len__()
    print('Test set size:', test_len)
    all_dataset = AffectNeutralImage(args.affect_label_path, args.affect_dataset_path, transform=data_transforms)
    all_loader = torch.utils.data.DataLoader(all_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)
    return train_loader, test_loader, all_loader
    # return train_dataset, test_dataset


def FERparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fer_label_path', type=str,
                        # default='../datasets/FERPLUS',
                        default='/media/database/data4/wf/GraphNet-FER/work01/datasets/FERPLUS',
                        help='FERPLUS label path.')
    parser.add_argument('--fer_dataset_path', type=str,
                        default='/media/database/data4/wf/GraphNet-FER/work01/datasets/FERPLUS',
                        help='FERPLUS dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


class FERNeutralImage(data.Dataset):
    def __init__(self, label_path, dataset_path, phase=None, transform=None):
        self.phase = phase
        self.transform = transform
        self.label_path = label_path
        self.dataset_path = dataset_path
        self.nodes = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger']

        ferplus = pd.read_csv(os.path.join(self.label_path, 'fer2013new.csv'))
        fer2013 = pd.read_csv(os.path.join(self.label_path, 'fer2013.csv'))[['emotion']]
        fer = pd.concat([ferplus, fer2013], axis=1)
        fer = fer.dropna(axis=0, how='any')  # 去除没有图片名字的行（没有名字的图片在读入时为nan）
        fer.columns = list(range(fer.shape[1]))
        # fer = fer.reset_index()
        fer1 = fer.iloc[:, :-1]  # ferplus
        fer1_1 = fer1.iloc[:, :2]  # ferplus的usage和image name
        fer1_2 = fer1.iloc[:, 2:]  # ferplus的投票结果
        fer2 = fer.iloc[:, -1]  # fer2013的标签
        row = fer1_1.loc[(fer1_2 == 0).all(axis=1), :]
        fer1_1 = fer1_1.loc[~(fer1_2 == 0).all(axis=1), :]  # 删除哪些在标签投票中哪个类别都没有票的图片
        fer1_2 = fer1_2.loc[~(fer1_2 == 0).all(axis=1), :]
        fer2 = fer2.drop(index=list(row.index))
        # fer2 = fer2.loc[~(fer1_2 == 0).all(axis=1), :]
        fer1_2.columns = list(range(fer1_2.shape[1]))  # 更新列名，表情类别名称与索引重新对应
        rowmax = fer1_2.idxmax(axis=1) - 1  # 获取每个图片投票情况中第一个出现的最大值
        fer1 = pd.concat([fer1_1, rowmax], axis=1)  # 拼接，得到usage、image name、class，便于后续处理
        fer = pd.concat([fer1, fer2], axis=1)
        fer.columns = list(range(fer.shape[1]))
        fer = fer[fer[2] == -1]
        self.fer = fer
        name_list = [0, 1, 3]
        LABEL_COLUMN = 2

        if phase == 'train':
            labels = []
            file_names = fer[name_list]
            self.file_paths = []
            for i in range(file_names.shape[0]):
                filei = file_names.iloc[i, :]
                usage = filei[0]
                if usage != 'PrivateTest':
                    name = int(filei[1].split('.')[0][3:])
                    raw_class = str(fer2013.iloc[name, 0])
                    path = os.path.join(self.dataset_path, usage, raw_class, str(name) + '.jpg')
                    self.file_paths.append(path)
                    labels.append(0)
            self.label = np.hstack(labels)

        elif phase == 'test':
            labels = []
            file_names = fer[name_list]
            self.file_paths = []
            for i in range(file_names.shape[0]):
                filei = file_names.iloc[i, :]
                usage = filei[0]
                if usage == 'PrivateTest':
                    name = int(filei[1].split('.')[0][3:])
                    raw_class = str(fer2013.iloc[name, 0])
                    path = os.path.join(self.dataset_path, usage, raw_class, str(name) + '.jpg')
                    self.file_paths.append(path)
                    labels.append(0)
            self.label = np.hstack(labels)

        else:
            labels = []
            file_names = fer[name_list]
            self.file_paths = []
            for i in range(file_names.shape[0]):
                filei = file_names.iloc[i, :]
                usage = filei[0]
                name = int(filei[1].split('.')[0][3:])
                raw_class = str(fer2013.iloc[name, 0])
                path = os.path.join(self.dataset_path, usage, raw_class, str(name) + '.jpg')
                self.file_paths.append(path)
                labels.append(0)
            self.label = np.hstack(labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path, 0)
        if image is None:
            image, label, index = None, None, None
            return image, label, index
        else:
            # image = image[:, :, ::-1]  # BGR to RGB
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.copy()
            label = self.label[idx]
            if self.transform is not None:
                image = self.transform(image)
        return image, label, idx


def getFERdata():
    args = FERparse_args()
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        transforms.Normalize([0.485, ], [0.229, ]),
        # transforms.RandomErasing(scale=(0.02, 0.25))
    ])
    train_dataset = FERNeutralImage(args.fer_label_path, args.fer_dataset_path,
                                       phase='train', transform=data_transforms)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    test_dataset = FERNeutralImage(args.fer_label_path, args.fer_dataset_path,
                                      phase='test', transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              shuffle=False,
                                              pin_memory=True)
    test_len = test_dataset.__len__()
    print('Test set size:', test_len)
    all_dataset = FERNeutralImage(args.fer_label_path, args.fer_dataset_path, transform=data_transforms)
    all_loader = torch.utils.data.DataLoader(all_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)
    return train_loader, test_loader, all_loader
    # return train_dataset, test_dataset

if __name__ == '__main__':
    train_loader, test_loader, all_loader = getFERdata()
