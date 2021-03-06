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


class RafSubDataSet(data.Dataset):
    def __init__(self, raf_path, label_index, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.nodes = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
        self.classes = len(self.nodes)

        NAME_COLUMN = 0
        LABEL_COLUMN = 1

        if phase == 'train':
            df = pd.read_csv(os.path.join(self.raf_path, 'basic/EmoLabel/list_patition_label.txt'), sep=' ',
                             header=None)
            row_list = df[df[LABEL_COLUMN] == 7].index.tolist()  # get the row of neutral and delete them
            # df = df.drop(row_list)
            df = df[df[NAME_COLUMN].str.startswith('train')]
            file_names = df.iloc[:, NAME_COLUMN].values
            self.label = df.iloc[:, LABEL_COLUMN].values - 1
            # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            self.file_paths = []
            # use raf aligned images for training/testing
            for f in file_names:
                f = f.split(".")[0]
                f = f + "_aligned.jpg"
                path = os.path.join(self.raf_path, 'basic/Image/aligned', f)
                self.file_paths.append(path)

        elif phase == 'test':
            df = pd.read_csv(os.path.join(self.raf_path, 'basic/EmoLabel/list_patition_label.txt'), sep=' ',
                             header=None)
            row_list = df[df[LABEL_COLUMN] == 7].index.tolist()  # get the row of neutral and delete them
            # df = df.drop(row_list)
            df = df[df[NAME_COLUMN].str.startswith('test')]
            df = df.reset_index().iloc[:, 1:]
            row_list = df[df[LABEL_COLUMN] == label_index+1].index.tolist()
            df = df.iloc[row_list]
            file_names = df.iloc[:, NAME_COLUMN].values
            self.label = df.iloc[:, LABEL_COLUMN].values - 1
            # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            self.file_paths = []
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
        # image = image[:, :, ::-1]  # BGR to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.copy()
        label = self.label[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx


def getRAFdata():
    args = parse_args()

    # ??????train data
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        transforms.Normalize([0.485, ], [0.229, ]),
        transforms.RandomErasing(scale=(0.02, 0.25))
    ])
    train_dataset = RafSubDataSet(args.raf_path, 0, phase='train', transform=data_transforms)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    test_dataset = []
    test_loader = []
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, ], [0.229, ]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    test_len = 0
    for i in range(len(train_dataset.nodes)):
        label_i = i
        test_dataset.append(RafSubDataSet(args.raf_path, label_i, phase='test',
                                          transform=data_transforms_val))
        test_loader.append(torch.utils.data.DataLoader(test_dataset[i],
                                                       batch_size=args.batch_size,
                                                       num_workers=args.workers,
                                                       shuffle=False,
                                                       pin_memory=True))
        test_len += test_dataset[i].__len__()
    print('Test set size:', test_len)

    return train_loader, test_loader
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


class AffectSubDataSet(data.Dataset):
    def __init__(self, label_path, dataset_path, label_index, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.label_path = label_path
        self.dataset_path = dataset_path
        self.nodes = [
            'Neutral',
            'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

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
        # row_list0 = df[df[LABEL_COLUMN] == 0].index.tolist()
        # df = df.drop(row_list0)
        row_list8 = df[df[LABEL_COLUMN] == 8].index.tolist()
        df = df.drop(row_list8)
        row_list9 = df[df[LABEL_COLUMN] == 9].index.tolist()
        df = df.drop(row_list9)
        row_list10 = df[df[LABEL_COLUMN] == 10].index.tolist()
        df = df.drop(row_list10)

        # 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt
        usage_column = 2
        df_tr = []
        df_te = []
        # for i in range(1, 8):
        for i in range(len(self.nodes)):
            dfi = df[df[LABEL_COLUMN] == i]
            dfi_train = dfi[dfi[usage_column] == 'training']
            dfi_val = dfi[dfi[usage_column] == 'validation']
            df_tr.append(dfi_train)
            df_te.append(dfi_val)

        if phase == 'train':
            file_names = []
            labels = []
            for tr_i in range(len(df_tr)):
                names = df_tr[tr_i].iloc[:, NAME_COLUMN].values
                file_names.append(names)
                label = np.array([tr_i] * names.shape[0])
                labels.append(label)
            file_names = np.hstack(file_names)
            self.label = np.hstack(labels)
            self.file_paths = []
            for f in file_names:
                f = f.split(".")[0]
                f = f + ".jpg"
                path = os.path.join(self.dataset_path, f)
                self.file_paths.append(path)

        if phase == 'test':
            file_names = []
            labels = []
            names = df_te[label_index].iloc[:, NAME_COLUMN].values
            file_names.append(names)
            label = np.array([label_index] * names.shape[0])
            labels.append(label)
            file_names = np.hstack(file_names)
            self.label = np.hstack(labels)
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
        # image = image[:, :, ::-1]  # BGR to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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


def getAffectdata():
    args = Affectparse_args()
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        transforms.Normalize([0.485, ], [0.229, ]),
        transforms.RandomErasing(scale=(0.02, 0.25))
    ])
    train_dataset = AffectSubDataSet(args.affect_label_path, args.affect_dataset_path,
                                     0, phase='train',
                                     transform=data_transforms)
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
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        transforms.Normalize([0.485, ], [0.229, ]),
    ])
    test_dataset = []
    test_loader = []
    test_len = 0
    for i in range(len(train_dataset.nodes)):
        test_dataset.append(AffectSubDataSet(args.affect_label_path, args.affect_dataset_path,
                                             i, phase='test',
                                             transform=data_transforms_val))
        test_loader.append(torch.utils.data.DataLoader(test_dataset[i],
                                                       batch_size=args.batch_size,
                                                       num_workers=args.workers,
                                                       shuffle=False,
                                                       pin_memory=True))
        test_len += test_dataset[i].__len__()
    print('Test set size:', test_len)
    return train_loader, test_loader
    # return train_dataset, test_dataset


def FERparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fer_label_path', type=str,
                        # default='../datasets/FERPLUS',
                        default='/media/database/data2/Expression/FERPLUS',
                        help='FERPLUS label path.')
    parser.add_argument('--fer_dataset_path', type=str,
                        default='/media/database/data2/Expression/FERPLUS',
                        help='FERPLUS dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


class FERSubDataSet(data.Dataset):
    def __init__(self, label_path, dataset_path, label_index, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.label_path = label_path
        self.dataset_path = dataset_path
        self.nodes = [
            'Neutral',
                      'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']

        ferplus = pd.read_csv(os.path.join(self.label_path, 'fer2013new.csv'))
        fer2013 = pd.read_csv(os.path.join(self.label_path, 'fer2013.csv'))[['emotion']]
        fer = pd.concat([ferplus, fer2013], axis=1)
        fer = fer.dropna(axis=0, how='any')  # ?????????????????????????????????????????????????????????????????????nan???
        fer.columns = list(range(fer.shape[1]))
        # fer = fer.reset_index()
        fer1 = fer.iloc[:, :-1]  # ferplus
        fer1_1 = fer1.iloc[:, :2]  # ferplus???usage???image name
        fer1_2 = fer1.iloc[:, 2:]  # ferplus???????????????
        fer2 = fer.iloc[:, -1]  # fer2013?????????
        '''
        row = fer1_1.loc[(fer1_2 == 0).all(axis=1), :]
        fer1_1 = fer1_1.loc[~(fer1_2 == 0).all(axis=1), :]  # ???????????????????????????????????????????????????????????????
        fer1_2 = fer1_2.loc[~(fer1_2 == 0).all(axis=1), :]
        fer2 = fer2.drop(index=list(row.index))
        # fer2 = fer2.loc[~(fer1_2 == 0).all(axis=1), :]
        '''
        fer1_2.columns = list(range(fer1_2.shape[1]))  # ??????????????????????????????????????????????????????
        rowmax = fer1_2.idxmax(axis=1) - 1  # ????????????????????????????????????????????????????????????
        fer1 = pd.concat([fer1_1, rowmax], axis=1)  # ???????????????usage???image name???class?????????????????????
        fer = pd.concat([fer1, fer2], axis=1)
        fer.columns = list(range(fer.shape[1]))
        # row_neutral = fer[fer[2] == -1].index.tolist()
        # fer = fer.drop(row_neutral)

        row_neutral = fer[fer[2] == -1].index.tolist()
        fer_neutral = fer.loc[row_neutral, :]
        row_happy = fer[fer[2] == 0].index.tolist()
        fer_happy = fer.loc[row_happy, :]
        row_surprise = fer[fer[2] == 1].index.tolist()
        fer_surprise = fer.loc[row_surprise, :]
        row_sadness = fer[fer[2] == 2].index.tolist()
        fer_sadness = fer.loc[row_sadness, :]
        row_anger = fer[fer[2] == 3].index.tolist()
        fer_anger = fer.loc[row_anger, :]
        row_disgust = fer[fer[2] == 4].index.tolist()
        fer_disgust = fer.loc[row_disgust, :]
        row_fear = fer[fer[2] == 5].index.tolist()
        fer_fear = fer.loc[row_fear, :]
        row_contempt = fer[fer[2] == 6].index.tolist()
        fer_contempt = fer.loc[row_contempt, :]
        self.fer = fer
        name_list = [0, 1, 3]
        LABEL_COLUMN = 2

        if phase == 'train':
            labeltp = []
            file_names = fer_neutral[name_list]
            file_names = pd.concat([file_names, fer_happy[name_list]], axis=0)
            file_names = pd.concat([file_names, fer_surprise[name_list]], axis=0)
            file_names = pd.concat([file_names, fer_sadness[name_list]], axis=0)
            file_names = pd.concat([file_names, fer_anger[name_list]], axis=0)
            file_names = pd.concat([file_names, fer_disgust[name_list]], axis=0)
            file_names = pd.concat([file_names, fer_fear[name_list]], axis=0)
            file_names = pd.concat([file_names, fer_contempt[name_list]], axis=0)

            labeltp.append([0] * fer_neutral.shape[0])
            labeltp.append([1] * fer_happy.shape[0])
            labeltp.append([2] * fer_surprise.shape[0])
            labeltp.append([3] * fer_sadness.shape[0])
            labeltp.append([4] * fer_anger.shape[0])
            labeltp.append([5] * fer_disgust.shape[0])
            labeltp.append([6] * fer_fear.shape[0])
            labeltp.append([7] * fer_contempt.shape[0])

            labeltp = np.hstack(labeltp)
            labels = []
            self.file_paths = []
            for i in range(file_names.shape[0]):
                filei = file_names.iloc[i, :]
                usage = filei[0]
                if usage != 'PrivateTest':
                    name = int(filei[1].split('.')[0][3:])
                    raw_class = str(fer2013.iloc[name, 0])
                    path = os.path.join(self.dataset_path, usage, raw_class, str(name) + '.jpg')
                    self.file_paths.append(path)
                    labels.append(labeltp[i])
            self.label = np.hstack(labels)

        if phase == 'test':
            labeltp = []
            if label_index == 0:
                file_names = fer_neutral[name_list]
                labeltp.append(fer_neutral.shape[0]*[0])
            elif label_index == 1:
                file_names = fer_happy[name_list]
                labeltp.append(fer_happy.shape[0]*[1])
            elif label_index == 2:
                file_names = fer_surprise[name_list]
                labeltp.append(fer_surprise.shape[0]*[2])
            elif label_index == 3:
                file_names = fer_sadness[name_list]
                labeltp.append(fer_sadness.shape[0]*[3])
            elif label_index == 4:
                file_names = fer_anger[name_list]
                labeltp.append(fer_anger.shape[0]*[4])
            elif label_index == 5:
                file_names = fer_disgust[name_list]
                labeltp.append(fer_disgust.shape[0]*[5])
            elif label_index == 6:
                file_names = fer_fear[name_list]
                labeltp.append(fer_fear.shape[0]*[6])
            elif label_index == 7:
                file_names = fer_contempt[name_list]
                labeltp.append(fer_contempt.shape[0]*[7])
            labeltp = np.hstack(labeltp)

            labels = []
            self.file_paths = []
            for i in range(file_names.shape[0]):
                filei = file_names.iloc[i, :]
                usage = filei[0]
                if usage == 'PrivateTest':
                    name = int(filei[1].split('.')[0][3:])
                    raw_class = str(fer2013.iloc[name, 0])
                    path = os.path.join(self.dataset_path, usage, raw_class, str(name) + '.jpg')
                    self.file_paths.append(path)
                    labels.append(labeltp[i])
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
        transforms.Grayscale(1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, ], [0.229, ]),
        # transforms.ColorJitter(brightness=0.5)
        transforms.RandomErasing(scale=(0.02, 0.25))
    ])
    train_dataset = FERSubDataSet(args.fer_label_path, args.fer_dataset_path,
                                  0, phase='train', transform=data_transforms)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, ], [0.229, ])
    ])
    test_dataset = []
    test_loader = []
    test_len = 0
    for i in range(len(train_dataset.nodes)):
        test_dataset.append(FERSubDataSet(args.fer_label_path, args.fer_dataset_path,
                                          i, phase='test', transform=data_transforms_val))
        test_loader.append(torch.utils.data.DataLoader(test_dataset[i],
                                                       batch_size=args.batch_size,
                                                       num_workers=args.workers,
                                                       shuffle=False,
                                                       pin_memory=True))
        test_len += test_dataset[i].__len__()
    print('Test set size:', test_len)
    return train_loader, test_loader
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


class CKSubDataset(data.Dataset):
    def __init__(self, data_path, label_index, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.data_path = os.path.join(data_path, 'ck+/processed')
        self.neutral_path = os.path.join(data_path, 'cohn-kanade-images')
        self.nodes = ['surprise', 'fear', 'disgust', 'happy', 'sadness', 'anger', 'neutral']
        self.classes = len(self.nodes)
        self.split = [0.9, 0.1]

        labels = []
        files = []
        for label, node in enumerate(self.nodes):
            if node != 'neutral':
                file_node = os.path.join(self.data_path, node)
                file_names = []
                img_files = []
                for home, dirs1, dirs2 in os.walk(file_node):
                    for tmp_file in dirs1:
                        img_files.append(os.path.join(home, tmp_file))
                    break
                for img_file_i in range(len(img_files)):
                    for home, dirs1, dirs2 in os.walk(img_files[img_file_i]):
                        for tmp_file in dirs2:
                            file_names.append(os.path.join(home, tmp_file))
                        break
                files.append(file_names)

            else:
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
                files.append(file_names)

        if phase == 'train':
            self.paths = []
            self.labels = []
            for class_i in range(len(self.nodes)):
                file_names = files[class_i]
                imgs_num = len(file_names)
                tr_num = int(imgs_num * self.split[0])
                for tr_img_i in range(tr_num):
                    self.paths.append(file_names[tr_img_i])
                    self.labels.append(class_i)

        elif phase == 'test':
            self.paths = []
            self.labels = []
            file_names = files[label_index]
            imgs_num = len(file_names)
            tr_num = int(imgs_num * self.split[0])
            for te_img_i in range(tr_num, imgs_num):
                self.paths.append(file_names[te_img_i])
                self.labels.append(label_index)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # image = image[:, :, ::-1]  # BGR to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            image, label, index = None, None, None
            return image, label, index
        else:
            # image = image[:, :, ::-1]  # BGR to RGB
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.copy()
            label = self.labels[idx]
            if self.transform is not None:
                image = self.transform(image)
        return image, label, idx


def getCKdata():
    args = CKparse_args()

    # ??????train data
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        transforms.Normalize([0.485, ], [0.229, ]),
        transforms.RandomErasing(scale=(0.02, 0.25))
    ])
    train_dataset = CKSubDataset(args.data_path, 0, phase='train', transform=data_transforms)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    test_dataset = []
    test_loader = []
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, ], [0.229, ]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    test_len = 0
    for i in range(len(train_dataset.nodes)):
        label_i = i
        test_dataset.append(CKSubDataset(args.data_path, label_i, phase='test',
                                          transform=data_transforms_val))
        test_loader.append(torch.utils.data.DataLoader(test_dataset[i],
                                                       batch_size=args.batch_size,
                                                       num_workers=args.workers,
                                                       shuffle=False,
                                                       pin_memory=True))
        test_len += test_dataset[i].__len__()
    print('Test set size:', test_len)

    # return train_dataset, test_dataset
    return train_loader, test_loader


if __name__ == '__main__':
    # RAF_train_loader, RAF_test_loader = getRAFdata()
    # AffectNet_train_loader, AffectNet_test_loader = getAffectdata()
    FER_train_loader, FER_test_loader = getFERdata()
    # CK_train_set, CK_test_set = getCKdata()