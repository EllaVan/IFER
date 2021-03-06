import argparse
import os.path as osp
import os
import sys
sys.path.append('../')

import datetime
import pytz

import torch
import torch.nn as nn
import numpy as np

from models.resnet import ResNet
from materials import image_feature_wN as get_dataloader


def set_gpu(gpu):
    torch.cuda.current_device()
    torch.cuda._initialized = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('using gpu {}'.format(gpu))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu choice')
    parser.add_argument('--dataset', type=str, default='SAMM')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_gpu(args.gpu)

    if args.dataset == 'RAF':
        train_loader, test_loader = get_dataloader.getRAFdata()
        res_file = torch.load('RAF/nN/epoch-40res.pth', map_location=torch.device('cpu'))
        fc_file = torch.load('RAF/nN/epoch-40fc.pth', map_location=torch.device('cpu'))
    elif args.dataset == 'AffectNet':
        train_loader, test_loader = get_dataloader.getAffectdata()
        res_file = torch.load('AffectNet/nN/epoch-30res.pth', map_location=torch.device('cpu'))
        fc_file = torch.load('AffectNet/nN/epoch-30fc.pth', map_location=torch.device('cpu'))
    elif args.dataset == 'FERPLUS':
        train_loader, test_loader = get_dataloader.getFERdata()
        res_file = torch.load('FERPLUS/wN/epoch-30res.pth', map_location=torch.device('cpu'))
        fc_file = torch.load('FERPLUS/wN/epoch-30fc.pth', map_location=torch.device('cpu'))
    elif args.dataset == 'CK+':
        train_loader, test_loader = get_dataloader.getCKdata()
        res_file = torch.load('CK+/wN_8/epoch-20res.pth', map_location=torch.device('cpu'))
        fc_file = torch.load('CK+/wN_8/epoch-20fc.pth', map_location=torch.device('cpu'))
    elif args.dataset == 'CASME2':
        train_loader, test_loader = get_dataloader.getCASME2data()
        res_file = torch.load('CASME2/wN/epoch-10res.pth', map_location=torch.device('cpu'))
        fc_file = torch.load('CASME2/wN/epoch-10fc.pth', map_location=torch.device('cpu'))
    elif args.dataset == 'SAMM':
        train_loader, test_loader = get_dataloader.getSAMMdata()
        res_file = torch.load('SAMM/wN/epoch-40res.pth', map_location=torch.device('cpu'))
        fc_file = torch.load('SAMM/wN/epoch-40fc.pth', map_location=torch.device('cpu'))

    nodes = train_loader.dataset.nodes
    num_cls = len(nodes)
    model = ResNet('resnet18', num_cls)
    model.resnet_base.load_state_dict(res_file, False)
    model.fc.load_state_dict(fc_file, False)
    model = model.cuda()
    model.eval()

    acc = []
    flag_not = 0

    with torch.no_grad():
        class_acc = []
        for label_i, label in enumerate(nodes):
            subloader = test_loader[label_i]
            for batch_i, (data, targets, indexes) in enumerate(subloader):
                if data is None:
                    flag_not += 1
                    continue
                else:
                    data = data.cuda()
                    targets = targets.cuda()

                    logits = model(data)
                    _, pred = torch.max(logits, dim=1)
                    acc += torch.eq(pred, targets).type(torch.FloatTensor)
                    class_acc += torch.eq(pred, targets).type(torch.FloatTensor)

            print('%s acc is %.4f' % (nodes[label_i], np.array(class_acc).mean()))
            class_acc = []

        print('acc of dataset: %.4f' % (np.array(acc).mean()))