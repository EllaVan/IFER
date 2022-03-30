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
        self.nodes = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

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
        # transforms.RandomErasing(scale=(0.02, 0.25))
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
    parser.add_argument('--fer_dataset_path', type=str,
                        default='/media/database/data2/Expression/FERPLUS/new/fer2013plus',
                        help='FERPLUS dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


class FERSubDataSet(data.Dataset):
    def __init__(self, dataset_path, phase, label_index=None, transform=None):
        self.phase = phase
        self.transform = transform
        self.dataset_path = dataset_path
        self.nodes = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']

        if phase == 'train':
            file_names = []
            labels = []
            self.dataset_path1 = os.path.join(dataset_path, 'train')
            self.dataset_path2 = os.path.join(dataset_path, 'valid')
            for i in range(len(self.nodes)):
                home_path1 = os.path.join(self.dataset_path1, str(i))
                home_path2 = os.path.join(self.dataset_path2, str(i))
                for home, dirs1, dirs2 in os.walk(home_path1):
                    for tmp_file in dirs2:
                        file_names.append(os.path.join(home, tmp_file))
                        labels.append(i)
                    break
                for home, dirs1, dirs2 in os.walk(home_path2):
                    for tmp_file in dirs2:
                        file_names.append(os.path.join(home, tmp_file))
                        labels.append(i)
                    break
            self.file_paths = file_names
            self.label = np.hstack(labels)

        elif phase == 'test':
            file_names = []
            labels = []
            self.dataset_path = os.path.join(dataset_path, 'test', str(label_index))
            for home, dirs1, dirs2 in os.walk(self.dataset_path):
                for tmp_file in dirs2:
                    file_names.append(os.path.join(home, tmp_file))
                    labels.append(label_index)
                break
            self.file_paths = file_names
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
        # transforms.RandomErasing(scale=(0.02, 0.25))
    ])
    train_dataset = FERSubDataSet(args.fer_dataset_path, phase='train', transform=data_transforms)
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
        test_dataset.append(FERSubDataSet(args.fer_dataset_path, phase='test',
                                          label_index=i, transform=data_transforms_val))
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
                        default='/media/database/data2/Expression/CK+')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


class CKSubDataset(data.Dataset):
    def __init__(self, data_path, label_index, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.data_path = os.path.join(data_path, 'ck+/processed')
        self.neutral_path = os.path.join(data_path, 'cohn-kanade-images')
        self.nodes = ['surprise', 'fear', 'disgust', 'happy', 'sadness', 'anger', 'contempt', 'neutral']
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
                neutral_files = []
                file_node = self.neutral_path
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
                        neutral_files.append(os.path.join(home, dirs2[0]))
                        neutral_files.append(os.path.join(home, dirs2[1]))
                        break
                files.append(neutral_files)

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


def CASME2parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/media/database/data2/Expression/CASME2')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


class CASME2SubDataset(data.Dataset):
    def __init__(self, data_path, label_index, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.label_path = os.path.join(data_path, 'CASME2-coding.xlsx')
        self.data_path = os.path.join(data_path, 'Cropped')
        label_file = pd.read_excel(self.label_path)
        self.nodes = ['sadness', 'repression', 'surprise', 'happiness', 'fear', 'disgust', 'others', 'neutral']
        self.emo_dict = dict(zip(self.nodes, list(range(len(self.nodes)))))
        self.split = [0.9, 0.1]

        onset_file_names = []
        offset_file_names = []
        emo_file_names = [], [], [], [], [], [], [], []
        for i in range(len(label_file)):
            cur_file_info = label_file.iloc[i]
            cur_file_path = os.path.join(self.data_path, 'sub'+'%02d' % int(cur_file_info['Subject']), cur_file_info['Filename'])

            emo_img_path = 'reg_img' + str(cur_file_info['ApexFrame']) + '.jpg'
            cur_emo_label = self.emo_dict[cur_file_info['Estimated Emotion']]
            # if cur_emo_label != len(self.nodes)-1:
            emo_file_names[cur_emo_label].append(os.path.join(cur_file_path, emo_img_path))

            onset_img_path = 'reg_img' + str(cur_file_info['OnsetFrame']) + '.jpg'
            offset_img_path = 'reg_img' + str(cur_file_info['OffsetFrame']) + '.jpg'
            emo_file_names[len(self.nodes)-1].append(os.path.join(cur_file_path, onset_img_path))
            emo_file_names[len(self.nodes)-1].append(os.path.join(cur_file_path, offset_img_path))

        if phase == 'train':
            # emo_file_names = emo_file_names + onset_file_names
            self.paths = []
            self.labels = []
            for class_i in range(len(self.nodes)):
                file_names = emo_file_names[class_i]
                imgs_num = len(file_names)
                tr_num = int(imgs_num * self.split[0])
                for tr_img_i in range(tr_num):
                    self.paths.append(file_names[tr_img_i])
                    self.labels.append(class_i)

        elif phase == 'test':
            # emo_file_names = emo_file_names + onset_file_names
            self.paths = []
            self.labels = []
            file_names = emo_file_names[label_index]
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


def getCASME2data():
    args = CASME2parse_args()

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
    train_dataset = CASME2SubDataset(args.data_path, 0, phase='train', transform=data_transforms)
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
        test_dataset.append(CASME2SubDataset(args.data_path, label_i, phase='test',
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


def SAMMparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/media/database/data2/Expression/SAMM/SAMM')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


class SAMMSubDataset(data.Dataset):
    def __init__(self, data_path, label_index, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.label_path = os.path.join(data_path, 'SAMM.xlsx')
        self.data_path = data_path
        label_file = pd.read_excel(self.label_path)
        # self.nodes = list(set(label_file['Estimated Emotion']))
        self.nodes = ['Fear', 'Happiness', 'Surprise', 'Contempt', 'Anger', 'Sadness', 'Disgust', 'Other', 'Neutral']
        self.emo_dict = dict(zip(self.nodes, list(range(len(self.nodes)))))
        self.split = [0.9, 0.1]

        emo_file_names = [], [], [], [], [], [], [], [], []
        for i in range(len(label_file)):
            cur_file_info = label_file.iloc[i]
            sub = '%03d' % int(cur_file_info['Subject'])
            cur_file_path = os.path.join(self.data_path, sub, cur_file_info['Filename'])
            format_len = len(os.listdir(cur_file_path)[0].split('.')[0].split('_')[1])
            format_len = '%0' + str(format_len) + 'd'
            emo_img_path = sub + '_' + format_len % int(cur_file_info['Apex Frame']) + '.jpg'
            cur_emo_label = self.emo_dict[cur_file_info['Estimated Emotion']]
            # if cur_emo_label != len(self.nodes) - 1:
            emo_file_names[cur_emo_label].append(os.path.join(cur_file_path, emo_img_path))

            onset_img_path = sub + '_' + format_len % int(cur_file_info['Onset Frame']) + '.jpg'
            offset_img_path = sub + '_' + format_len % int(cur_file_info['Offset Frame']) + '.jpg'
            emo_file_names[len(self.nodes) - 1].append(os.path.join(cur_file_path, onset_img_path))
            emo_file_names[len(self.nodes) - 1].append(os.path.join(cur_file_path, offset_img_path))

        if phase == 'train':
            # emo_file_names = emo_file_names + onset_file_names
            self.paths = []
            self.labels = []
            for class_i in range(len(self.nodes)):
                file_names = emo_file_names[class_i]
                imgs_num = len(file_names)
                tr_num = int(imgs_num * self.split[0])
                for tr_img_i in range(tr_num):
                    self.paths.append(file_names[tr_img_i])
                    self.labels.append(class_i)

        elif phase == 'test':
            # emo_file_names = emo_file_names + offset_file_names
            self.paths = []
            self.labels = []
            file_names = emo_file_names[label_index]
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


def getSAMMdata():
    args = SAMMparse_args()

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
    train_dataset = SAMMSubDataset(args.data_path, 0, phase='train', transform=data_transforms)
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
        test_dataset.append(SAMMSubDataset(args.data_path, label_i, phase='test',
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
    # RAF_train_loader, RAF_test_loader, RAF_all_loader = getRAFdata()
    # AffectNet_train_loader, AffectNet_test_loader, AffectNet_all_loader = getAffectdata()
    # FER_train_loader, FER_test_loader, FER_all_loader = getFERdata()
    # CK_train_set, CK_test_set, CK_all_loader = getCKdata()
    CASME2_train_set, CASME2_test_set = getCASME2data()
    SAMM_train_set, SAMM_test_set = getSAMMdata()