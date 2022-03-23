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
                        default='/media/database/data2/Expression/RAF-DB',
                        # default='E:/DataSet/RAF-DB',
                        help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


class RAFNeutralImage(data.Dataset):
    def __init__(self, raf_path, phase=None, transform=None):
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
        elif phase == 'all':
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
            mot = torch.normal(0, 0.2, (1, 512))
            self.noise[idx] = mot

        return image, mot, idx


def getRAFdata():
    args = parse_args()

    # 加载train data
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, ], [0.229, ]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = RAFNeutralImage(args.raf_path, phase='train', transform=data_transforms)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    test_dataset = RAFNeutralImage(args.raf_path, phase='test', transform=data_transforms)
    print('Test set size:', test_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)
    all_dataset = RAFNeutralImage(args.raf_path, phase='all', transform=data_transforms)
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
                        default='/media/database/data2/Expression/AffectNet/label',
                        # default='../../datasets/AffectNet/Manually_Annotated_file_lists',
                        help='AffectNet label path.')
    parser.add_argument('--affect_dataset_path', type=str,
                        default='/media/database/data2/Expression/AffectNet/data/AffectNet',
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

        elif phase == 'all':
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
                mot = torch.normal(0, 0.2, (1, 512))
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
    all_dataset = AffectNeutralImage(args.affect_label_path, args.affect_dataset_path,
                                     phase='all', transform=data_transforms)
    all_loader = torch.utils.data.DataLoader(all_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=True,
                                             pin_memory=True)
    all_len = all_loader.__len__()
    print('All set size:', all_len)
    return train_loader, test_loader, all_loader
    # return train_dataset, test_dataset


def FERparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fer_dataset_path', type=str,
                        default='/media/database/data2/Expression/FERPLUS/new/fer2013plus',
                        help='FERPLUS dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


class FERNeutralImage(data.Dataset):
    def __init__(self, dataset_path, phase=None, transform=None):
        self.noise = {}
        self.phase = phase
        self.transform = transform
        self.dataset_path = dataset_path

        if phase == 'train':
            file_names = []
            self.dataset_path1 = os.path.join(dataset_path, 'train')
            self.dataset_path2 = os.path.join(dataset_path, 'valid')
            home_path1 = os.path.join(self.dataset_path1, '0')
            home_path2 = os.path.join(self.dataset_path2, '0')
            for home, dirs1, dirs2 in os.walk(home_path1):
                for tmp_file in dirs2:
                    file_names.append(os.path.join(home, tmp_file))
                break
            for home, dirs1, dirs2 in os.walk(home_path2):
                for tmp_file in dirs2:
                    file_names.append(os.path.join(home, tmp_file))
                break
            self.file_paths = file_names

        elif phase == 'test':
            file_names = []
            labels = []
            self.dataset_path = os.path.join(dataset_path, 'test', '0')
            for home, dirs1, dirs2 in os.walk(self.dataset_path):
                for tmp_file in dirs2:
                    file_names.append(os.path.join(home, tmp_file))
                break
            self.file_paths = file_names

        elif phase == 'all':
            file_names = []
            self.dataset_path1 = os.path.join(dataset_path, 'train')
            self.dataset_path2 = os.path.join(dataset_path, 'valid')
            self.dataset_path3 = os.path.join(dataset_path, 'test')
            home_path1 = os.path.join(self.dataset_path1, '0')
            home_path2 = os.path.join(self.dataset_path2, '0')
            home_path3 = os.path.join(self.dataset_path2, '0')
            for home, dirs1, dirs2 in os.walk(home_path1):
                for tmp_file in dirs2:
                    file_names.append(os.path.join(home, tmp_file))
                break
            for home, dirs1, dirs2 in os.walk(home_path2):
                for tmp_file in dirs2:
                    file_names.append(os.path.join(home, tmp_file))
                break
            for home, dirs1, dirs2 in os.walk(home_path3):
                for tmp_file in dirs2:
                    file_names.append(os.path.join(home, tmp_file))
                break
            self.file_paths = file_names

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
                mot = torch.normal(0, 0.2, (1, 512))
                self.noise[idx] = mot
        return image, mot, idx


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
    train_dataset = FERNeutralImage(args.fer_dataset_path, phase='train', transform=data_transforms)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    test_dataset = FERNeutralImage(args.fer_dataset_path, phase='test', transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              shuffle=False,
                                              pin_memory=True)
    test_len = test_dataset.__len__()
    print('Test set size:', test_len)
    all_dataset = FERNeutralImage(args.fer_dataset_path, phase='all', transform=data_transforms)
    all_loader = torch.utils.data.DataLoader(all_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=True,
                                             pin_memory=True)
    all_len = all_loader.__len__()
    print('All set size:', all_len)
    return train_loader, test_loader, all_loader
    # return train_dataset, test_dataset


def CKparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/media/database/data2/Expression/CK+',
                        # default='E:/DataSet/RAF-DB',
                        help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


class CKNeutralImage(data.Dataset):
    def __init__(self, data_path, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.neutral_path = os.path.join(data_path, 'cohn-kanade-images')
        self.nodes = ['surprise', 'fear', 'disgust', 'happy', 'sadness', 'anger', 'neutral']
        self.classes = len(self.nodes)
        self.split = [0.9, 0.1]
        self.noise = {}

        labels = []
        files = []
        file_node = self.neutral_path
        file_names = []
        subject_file = []
        video_file = []
        for home, dirs1, dirs2 in os.walk(file_node):
            for each_file in dirs1:
                subject_file.append(os.path.join(home, each_file))
            break
        for subject_i in range(len(subject_file)):
            for home, dirs1, dirs2 in os.walk(subject_file[subject_i]):
                for each_file in dirs1:
                    video_file.append(os.path.join(home, each_file))
                break

        for video_i in range(len(video_file)):
            for home, dirs1, dirs2 in os.walk(video_file[video_i]):
                file_names.append(os.path.join(home, dirs2[0]))
                file_names.append(os.path.join(home, dirs2[1]))
                break
        files = file_names
        imgs_num = len(file_names)
        tr_num = int(imgs_num * self.split[0])

        if phase == 'train':
            self.paths = []
            for tr_img_i in range(tr_num):
                self.paths.append(files[tr_img_i])

        elif phase == 'test':
            self.paths = []
            for te_img_i in range(tr_num, imgs_num):
                self.paths.append(files[te_img_i])

        elif phase == 'all':
            self.paths = files

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
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
                mot = torch.normal(0, 0.2, (1, 512))
                self.noise[idx] = mot
        return image, mot, idx


def getCKdata():
    args = CKparse_args()

    # 加载train data
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        transforms.Normalize([0.485, ], [0.229, ]),
        # transforms.RandomErasing(scale=(0.02, 0.25))
    ])
    train_dataset = CKNeutralImage(args.data_path, phase='train', transform=data_transforms)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, ], [0.229, ]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    test_dataset = CKNeutralImage(args.data_path, phase='test',
                                      transform=data_transforms_val)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True)
    test_len = test_dataset.__len__()
    print('Test set size:', test_len)

    all_dataset = CKNeutralImage(args.data_path, phase='all',
                                  transform=data_transforms)
    all_loader = torch.utils.data.DataLoader(all_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              shuffle=False,
                                              pin_memory=True)
    all_len = all_dataset.__len__()
    print('All set size:', all_len)

    return train_loader, test_loader, all_loader


if __name__ == '__main__':
    train_loader, test_loader, all_loader = getCKdata()
