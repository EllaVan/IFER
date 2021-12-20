import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
import torch
import os
import argparse

from materials.glove import GloVe
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str,
                        default='/media/database/data4/wf/work01/datasets/RAF-DB',
                        # default='E:/DataSet/RAF-DB',
                        help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--test_num', type=int, default=11)
    return parser.parse_args()


class RAFNeutralImage(data.Dataset):
    def __init__(self, raf_path, label_index, phase=None, transform=None):
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
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.copy()
        if self.transform is not None:
            image = self.transform(image)

        return image, idx


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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(scale=(0.02, 0.25))
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

# train_loader, test_loader, all_loader = getRAFdata()
