import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as mp, seaborn
import os.path as osp
import time

from config import opt
from materials import neutral_image, image_feature
import util
from models import model, dis_model
from models.resnet import make_resnet18_base
from models.resnetDeconv import make_resnet18_deconv_base

from PIL import Image
from torchvision.transforms import ToPILImage

import scipy.stats as stats
import scipy.optimize as opt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manualSeed', type=int, default=20, help='manual seed')
    parser.add_argument('--continue_from', type=int, default=0, help='save interval')
    parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=50, help='number of epochs to save for')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--size_mot', type=int, default=512, help='dimension of motion')
    parser.add_argument('--size_att', type=int, default=512, help='dimension of attribute')
    parser.add_argument('--size_res', type=int, default=512, help='dimension of image feature')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[512, 1024])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 512])

    parser.add_argument("--encoder_use_mot", default=False, help="Encoder use motion as input")

    return parser.parse_args()
args = parse_args()


global device
# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

f = torch.load('/media/database/data4/wf/IFER/codes2/mot_gaus_v2/1123/net_state/epoch1700')
cnn = make_resnet18_base()
sd = cnn.state_dict()
sd.update(f['cnn'])
cnn.load_state_dict(sd)
cnn.to(device)

cnn_deconv = make_resnet18_deconv_base()
sd = cnn_deconv.state_dict()
sd.update(f['cnn_deconv'])
cnn_deconv.load_state_dict(sd)
cnn_deconv.to(device)

netE1 = dis_model.Encoder1(args)
sd = netE1.state_dict()
sd.update(f['netE1'])
netE1.load_state_dict(sd)
netE1.to(device)

netE2 = dis_model.Encoder2(args).to(device)
# sd = netE2.state_dict()
# sd.update(f['netE2'])
# netE2.load_state_dict(sd)
# netE2.to(device)

netG_mot = dis_model.Generator_mot(args)
sd = netG_mot.state_dict()
sd.update(f['netG_mot'])
netG_mot.load_state_dict(sd)
netG_mot.to(device)

train_loader, test_loader = image_feature.getRAFdata()
test_nodes = train_loader.dataset.test_nodes

means = (f['means']).to(device)
log_var = (f['log_var']).to(device)
std = torch.exp(0.5 * log_var)
# std = log_var
channel_mean=[0.485, 0.456, 0.406]
channel_std=[0.229, 0.224, 0.225]

with torch.no_grad():
    for label_i, label in enumerate(test_nodes):
        subloader = test_loader[2]
        for batch_i, (imgs, targets, indexes) in enumerate(subloader):
            cos_att = torch.zeros(imgs.size(0), imgs.size(0))
            l2_att = torch.zeros(imgs.size(0), imgs.size(0))
            cos_att_eps = torch.zeros(imgs.size(0), imgs.size(0))
            l2_att_eps = torch.zeros(imgs.size(0), imgs.size(0))

            cos_mot = torch.zeros(imgs.size(0), imgs.size(0))
            l2_mot = torch.zeros(imgs.size(0), imgs.size(0))
            cos_mot_eps = torch.zeros(imgs.size(0), imgs.size(0))
            l2_mot_eps = torch.zeros(imgs.size(0), imgs.size(0))

            cos_res = torch.zeros(imgs.size(0), imgs.size(0))
            l2_res = torch.zeros(imgs.size(0), imgs.size(0))

            cos_eps = torch.zeros(imgs.size(0), imgs.size(0))
            l2_eps = torch.zeros(imgs.size(0), imgs.size(0))

            stat_vals = np.zeros((imgs.size(0), imgs.size(0)))
            p_vals = np.zeros((imgs.size(0), imgs.size(0)))


            imgs = imgs.to(device)
            res, indices = cnn(imgs)
            res_numpy = res.cpu().detach().numpy()

            # means, log_var = netE1(input_att)

            mot = netG_mot(res)
            mot_numpy = mot.cpu().detach().numpy()

            for i in range(imgs.size(0)):
                for j in range(i+1, imgs.size(0)):
                    cos_mot[i][j] = torch.cosine_similarity(mot[i].reshape(-1, 512), mot[j].reshape(-1, 512))
                    l2_mot[i][j] = torch.sqrt(torch.sum(mot[i] - mot[j])**2)
                    stat_vals[i][j], p_vals[i][j] = stats.ttest_rel(res_numpy[i], mot_numpy[j])

            cos_mot = cos_mot.cpu().detach().numpy()
            l2_mot = l2_mot.cpu().detach().numpy()

            break
        break

