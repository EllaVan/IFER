import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

import os


def draw_recon(x, x_recon):
    x_l, x_recon_l = x.tolist(), x_recon.tolist()
    result = [None] * (len(x_l) + len(x_recon_l))
    result[::2] = x_l
    result[1::2] = x_recon_l
    return torch.FloatTensor(result)


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')


def make_dataloader(args):

    test_loader = None
    train_loader = None
    if args.dataset == 'celeba':

        trans_f = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CelebA(args.data_dir, split='train', download=False, transform=trans_f)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                                   drop_last=True, num_workers=4)

    elif 'pendulum' in args.dataset:
        # to be released
        pass

    return train_loader, test_loader


def check_for_CUDA(sagan_obj):
    if not sagan_obj.config.disable_cuda and torch.cuda.is_available():
        print("CUDA is available!")
        sagan_obj.device = torch.device('cuda')
        sagan_obj.config.dataloader_args['pin_memory'] = True
    else:
        print("Cuda is NOT available, running on CPU.")
        sagan_obj.device = torch.device('cpu')

    if torch.cuda.is_available() and sagan_obj.config.disable_cuda:
        print("WARNING: You have a CUDA device, so you should probably run without --disable_cuda")


def prior_AUs(num_AU):
    A = torch.zeros((num_AU, num_AU))
    AU_name = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10',
               'AU12', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26']
    num_AU = len(AU_name)
    AU_index = list(range(num_AU))
    AU_dict = dict(zip(AU_index, AU_name))
    A[0, 1] = 1
    A[2, 5] = 1
    A[7, 4] = 1
    A[4, 5] = 1
    A[4, 8] = 1
    A[9, 13] = 1
    A[10, 13] = 1
    A[13, 12] = 1
    return AU_index, AU_name, AU_dict, A
