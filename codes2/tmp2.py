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

f = torch.load('/media/database/data4/wf/IFER/codes2/mot_gaus_v2/1130/net_state/epoch1100')
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
sd = netE2.state_dict()
sd.update(f['netE2'])
netE2.load_state_dict(sd)
netE2.to(device)

netG_mot = dis_model.Generator_mot(args)
sd = netG_mot.state_dict()
sd.update(f['netG_mot'])
netG_mot.load_state_dict(sd)
netG_mot.to(device)

train_loader, test_loader = image_feature.getRAFdata()
test_nodes = train_loader.dataset.test_nodes

means = (f['means']).to(device)
log_var = (f['log_var']).to(device)
# std = torch.exp(0.5 * log_var)
std = log_var
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

            eps = netE2(res)
            eps_numpy = eps.detach().cpu().numpy()
            att_eps = means.repeat(imgs.size(0), 1) + eps*std
            mot_eps = netG_mot(res, att_eps)
            mot_eps_numpy = mot_eps.cpu().detach().numpy()

            att = means.repeat(imgs.size(0), 1)
            mot = netG_mot(res, att)

            mot_att_is_eps = netG_mot(res, eps)

            # ---------------att_eps可视化------------------------
            recon_data1 = att_eps.unsqueeze(2)
            recon_data2 = recon_data1.expand(recon_data1.size(0), recon_data1.size(1), 49)
            recon_data3 = recon_data2.reshape(recon_data1.size(0), recon_data1.size(1), 7, 7)
            torch.cuda.empty_cache()
            recon_data_att_eps = cnn_deconv(recon_data3, indices)
            #---------------att_eps可视化------------------------

            # ---------------eps可视化------------------------
            eps_img1 = eps.unsqueeze(2)
            eps_img2 = eps_img1.expand(eps_img1.size(0), eps_img1.size(1), 49)
            eps_img3 = eps_img2.reshape(eps_img1.size(0), eps_img1.size(1), 7, 7)
            torch.cuda.empty_cache()
            recon_data_eps = cnn_deconv(eps_img3, indices)
            # ---------------eps可视化------------------------

            # for image_i in range(int(imgs.size(0)/2)):
            #     # deconv_img = a[image_i]
            #     # ori_img = data[image_i]
            #     for channel_i in range(3):
            #         recon_data_att_eps[image_i][channel_i, :, :] = recon_data_att_eps[image_i][channel_i, :, :] * channel_std[
            #             channel_i] + channel_mean[channel_i]
            #         recon_data_eps[image_i][channel_i, :, :] = recon_data_eps[image_i][channel_i, :, :] * \
            #                                                        channel_std[channel_i] + channel_mean[channel_i]
            #     att_eps_img = ToPILImage()(recon_data_att_eps[image_i])
            #     eps_img = ToPILImage()(recon_data_eps[image_i])
            #     vis_name = subloader.dataset.file_paths[indexes[image_i].detach().cpu().item()]
            #     vis_name = vis_name.split('.')[0].split('/')[-1]
            #     att_eps_path = os.path.join('mot_gaus_v2/1130/test_vis', vis_name + '_att_eps.png')
            #     eps_path = os.path.join('mot_gaus_v2/1130/test_vis', vis_name + '_eps.png')
            #     att_eps_img.save(att_eps_path)
            #     eps_img.save(eps_path)



            for i in range(imgs.size(0)):
                for j in range(i+1, imgs.size(0)):
                    cos_att[i][j] = torch.cosine_similarity(att[i].reshape(-1, 512), att[j].reshape(-1, 512))
                    l2_att[i][j] = torch.sqrt(torch.sum(att[i] - att[j])**2)
                    cos_att_eps[i][j] = torch.cosine_similarity(att_eps[i].reshape(-1, 512), att_eps[j].reshape(-1, 512))
                    l2_att_eps[i][j] = torch.sqrt(torch.sum(att_eps[i] - att_eps[j])**2)
                    cos_mot[i][j] = torch.cosine_similarity(mot[i].reshape(-1, 512), mot[j].reshape(-1, 512))
                    l2_mot[i][j] = torch.sqrt(torch.sum(mot[i] - mot[j])**2)
                    cos_mot_eps[i][j] = torch.cosine_similarity(mot_eps[i].reshape(-1, 512), mot_eps[j].reshape(-1, 512))
                    l2_mot_eps[i][j] = torch.sqrt(torch.sum(mot_eps[i] - mot_eps[j]) ** 2)
                    cos_res[i][j] = torch.cosine_similarity(res[i].reshape(-1, 512), res[j].reshape(-1, 512))
                    l2_res[i][j] = torch.sqrt(torch.sum(res[i] - res[j])**2)
                    cos_eps[i][j] = torch.cosine_similarity(eps[i].reshape(-1, 512), eps[j].reshape(-1, 512))
                    l2_eps[i][j] = torch.sqrt(torch.sum((eps[i]-eps[j])**2))
                    stat_vals[i][j], p_vals[i][j] = stats.ttest_rel(res_numpy[i], mot_eps_numpy[j])


            cos_att = cos_att.cpu().detach().numpy()
            l2_att = l2_att.cpu().detach().numpy()

            cos_att_eps = cos_att_eps.cpu().detach().numpy()
            l2_att_eps = l2_att_eps.cpu().detach().numpy()

            cos_mot = cos_mot.cpu().detach().numpy()
            l2_mot = l2_mot.cpu().detach().numpy()

            cos_mot_eps = cos_mot_eps.cpu().detach().numpy()
            l2_mot_eps = l2_mot_eps.cpu().detach().numpy()

            cos_res = cos_res.cpu().detach().numpy()
            l2_res = l2_res.cpu().detach().numpy()

            cos_mot_res_differ = cos_mot - cos_res
            l2_mot_res_differ = l2_mot - l2_res

            cos_mot_res_eps_differ = cos_mot_eps - cos_res
            l2_mot_res_eps_differ = l2_mot_eps - l2_res

            cos_eps = cos_eps.cpu().detach().numpy()
            l2_eps = l2_eps.cpu().detach().numpy()

            # print('mean of sim_mot_eps', np.mean(sim_mot_eps))
            # print('mean of sim_res', np.mean(sim_res))
            # stat_val, p_val = stats.ttest_ind(sim_mot_eps, sim_res, equal_var=False)

            # seaborn.heatmap(cos_mot_res_eps_differ, center=0, annot=False, xticklabels=list(range(imgs.size(0))),
            #                 yticklabels=list(range(imgs.size(0))))
            # mp.title('cos_mot_res_eps_differ')
            # mp.show()
            # seaborn.heatmap(l2_mot_res_eps_differ, center=0, annot=False, xticklabels=list(range(imgs.size(0))),
            #                 yticklabels=list(range(imgs.size(0))))
            # mp.title('l2_mot_res_eps_differ')
            # mp.show()
            # mp.close()

            break
        break

