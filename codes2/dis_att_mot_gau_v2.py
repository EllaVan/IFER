import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import time
import pytz
import datetime

# from config import opt
from materials import neutral_image
import util
from models import dis_model
from models.resnet import make_resnet18_base
from models.resnetDeconv import make_resnet18_deconv_base

# from lib import WeightedL1, calc_gradient_penalty, loss_fn, mse_loss, contrastive_loss


def mse_loss_att_res(att, res):
    # x_recon = model.conditional_sample(X, S, deterministic=deterministic)
    mse_loss_val = nn.MSELoss()(att, res)
    return mse_loss_val

def mse_loss_att_att(att1, att2):
    # x_recon = model.conditional_sample(X, S, deterministic=deterministic)
    mse_loss_val = nn.MSELoss()(att1, att2)
    return mse_loss_val

def mse_loss_mot_mot(mot1, mot2):
    # x_recon = model.conditional_sample(X, S, deterministic=deterministic)
    mse_loss_val = nn.MSELoss()(mot1, mot2)
    return mse_loss_val

def mse_loss_res_res(res1, res2):
    # x_recon = model.conditional_sample(X, S, deterministic=deterministic)
    mse_loss_val = nn.MSELoss()(res1, res2)
    return mse_loss_val

def mse_loss_fig_fig(fig1, fig2):
    # x_recon = model.conditional_sample(X, S, deterministic=deterministic)
    mse_loss_val = nn.MSELoss()(fig1, fig2)
    return mse_loss_val


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='cuda:1', help='gpu choice')
    parser.add_argument('--manualSeed', type=int, default=20, help='manual seed')
    parser.add_argument('--continue_from', type=int, default=0, help='save interval')
    parser.add_argument('--log_dir', type=str, default="mot_gaus_v2/1130", help='log_dir')
    parser.add_argument('--continue_file', type=str,
                        default='/media/database/data4/wf/IFER/codes2/mot_gaus_v3/1123/net_state/epoch200',
                        help='net state file')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=100, help='number of epochs to save for')

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

# print('Current Training time', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')))
torch.cuda.empty_cache()

# gpu设置
global device
# device = torch.device('cpu')
device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.manualSeed)

train_loader, test_loader, all_loader = neutral_image.getRAFdata()

# resnet18提取图片特征
cnn = make_resnet18_base()
sd = cnn.state_dict()
sd.update(torch.load('materials/resnet18-base.pth'))
cnn.load_state_dict(sd)
cnn.to(device)
cnn_deconv = make_resnet18_deconv_base()
cnn_deconv.to(device)

netE1 = dis_model.Encoder1(args).to(device)
netE2 = dis_model.Encoder2(args).to(device)# 编码器
netG = dis_model.Generator(args).to(device)    # 生成器
netG_mot = dis_model.Generator_mot(args).to(device)

if args.continue_from != 0:
    f = torch.load(args.continue_file)
    cnn.load_state_dict(f['cnn'])
    cnn_deconv.load_state_dict(f['cnn_deconv'])
    netE1.load_state_dict(f['netE1'])
    netE2.load_state_dict(f['netE2'])
    netG.load_state_dict(f['netG'])
    netG_mot.load_state_dict(f['netG_mot'])

optimizerE1 = optim.Adam(netE1.parameters(), lr=args.lr)
optimizerE2 = optim.Adam(netE2.parameters(), lr=args.lr)
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
# optimizerCNN = optim.SGD(cnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
optimizerCNN = optim.Adam(cnn.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG_mot = optim.Adam(netG_mot.parameters(), lr=args.lr*10, betas=(args.beta1, 0.999))
optimizerCnnDeconv = optim.Adam(cnn_deconv.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
# optimizerCnnDeconv = optim.SGD(cnn_deconv.parameters(), lr=0.0001,
#                       momentum=0.9,
#                       weight_decay=5e-4,
#                       nesterov=True)

log_dir = args.log_dir
writer = SummaryWriter(log_dir)
folders = ['data_vis', 'mean_vis', 'test_vis', 'net_state']
for folder in folders:
    if not os.path.exists(os.path.join(log_dir, folder)):
        os.mkdir(os.path.join(log_dir, folder))

channel_mean=[0.485, 0.456, 0.406]
channel_std=[0.229, 0.224, 0.225]

for epoch in range(args.continue_from, args.nepoch):
    torch.cuda.empty_cache()

    if epoch < 300:
        cnn.train()
        cnn_deconv.train()
        netG.train()
        netG_mot.train()
        netE1.train()
        netE2.train()
    else:
        cnn.eval()
        cnn_deconv.train()
        netG.eval()
        netG_mot.train()
        netE1.eval()
        netE2.eval()

    flag_not = 0
    train_size = 0

    loss1_float = []
    loss2_float = []
    loss3_float = []
    loss4_float = []
    loss5_float = []
    loss6_float = []
    loss_246_float = []

    means_record_sum_list = []
    log_var_record_sum_list = []
    fake_means_record_sum_list = []
    fake_log_var_record_sum_list = []
    means2_record_sum_list = []
    log_var2_record_sum_list = []
    fake_means2_record_sum_list = []
    fake_log_var2_record_sum_list = []

    pool_indices = []

    with torch.autograd.set_detect_anomaly(True):
        for i, (data, index) in enumerate(all_loader, 1):
            torch.cuda.empty_cache()
            for p in cnn.parameters():
                p.requires_grad = True
            for p in netE1.parameters():
                p.requires_grad = True
            for p in netE2.parameters():
                p.requires_grad = True
            for p in netG.parameters():
                p.requires_grad = True
            for p in netG_mot.parameters():
                p.requires_grad = True
            for p in cnn_deconv.parameters():
                p.requires_grad = True

            if data is None:
                flag_not += 1
                continue
            else:
                optimizerG.zero_grad()
                optimizerG_mot.zero_grad()
                optimizerE1.zero_grad()
                optimizerE1.zero_grad()
                optimizerCNN.zero_grad()

                train_size = train_size + data.size(0)
                data = data.to(device)
                input_att, indices = cnn(data) # resnet18生成的图片特征被认为是人脸属性特征
                pool_indices.append(indices)
                noise = torch.randn(data.size(0), args.size_mot).to(device)  # 随机生成一个标准正态分布的noise
                input_mot = noise # noise被认为是中性表情的动作特征
                input_res = netG(att=input_att, mot=input_mot)

                means, log_var = netE1(input_att)
                means_record_sum_list.append(torch.sum(means, dim=0, keepdim=True))
                means_record = torch.sum(torch.vstack(means_record_sum_list), dim=0, keepdim=True)
                means_record = Variable(means_record / train_size)
                log_var_record_sum_list.append(torch.sum(log_var, dim=0, keepdim=True))
                log_var_record = torch.sum(torch.vstack(log_var_record_sum_list), dim=0, keepdim=True)
                log_var_record = Variable(log_var_record / train_size)
                std = torch.exp(0.5 * log_var_record)
                # std = log_var_record
                eps = netE2(input_res)
                z = means_record.repeat(data.size(0), 1) + eps * std

                fake_res = netG(z, input_mot)

                fake_means, fake_log_var = netE1(z)
                fake_means_record_sum_list.append(torch.sum(fake_means, dim=0, keepdim=True))
                fake_means_record = torch.sum(torch.vstack(fake_means_record_sum_list), dim=0, keepdim=True)
                fake_means_record = Variable(fake_means_record / train_size)
                fake_log_var_record_sum_list.append(torch.sum(fake_log_var, dim=0, keepdim=True))
                fake_log_var_record = torch.sum(torch.vstack(fake_log_var_record_sum_list), dim=0, keepdim=True)
                fake_log_var_record = Variable(fake_log_var_record / train_size)
                # fake_std = torch.exp(0.5 * fake_log_var_record)
                fake_std = fake_log_var_record
                fake_eps = netE2(fake_res)
                fake_z = fake_means_record.repeat(data.size(0), 1) + fake_eps * fake_std

                fake_mot1 = netG_mot(input_res, z)
                fake_mot2 = netG_mot(input_res, input_att)

                # loss1 = mse_loss_att_res(input_att, input_res)
                # loss1_float.append(util.loss_to_float(loss1))
                loss2 = mse_loss_att_att(z, input_att)
                loss2_float.append(util.loss_to_float(loss2))
                # loss3 = mse_loss_res_res(input_res, fake_res)
                # loss3_float.append(util.loss_to_float(loss3))
                # loss4 = mse_loss_att_att(z, fake_z)
                # loss4_float.append(util.loss_to_float(loss4))
                loss4 = mse_loss_att_att(z, fake_z)
                loss4_float.append(util.loss_to_float(loss4))
                loss6 = mse_loss_mot_mot(fake_mot2, input_mot)#+mse_loss_mot_mot(fake_mot2, input_mot)+mse_loss_mot_mot(fake_mot1, fake_mot2)
                loss6_float.append(util.loss_to_float(loss6))

                loss_246 = loss2 + loss4 + loss6
                loss_246_float.append(util.loss_to_float(loss_246))
                loss_246.backward()
                optimizerE1.step()
                optimizerE2.step()
                optimizerG.step()
                optimizerG_mot.step()
                optimizerCNN.step()

                torch.cuda.empty_cache()
                optimizerCnnDeconv.zero_grad()

                for p in cnn.parameters():
                    p.requires_grad = False
                for p in netE1.parameters():
                    p.requires_grad = False
                for p in netE2.parameters():
                    p.requires_grad = False
                for p in netG.parameters():
                    p.requires_grad = False
                for p in netG_mot.parameters():
                    p.requires_grad = False

                input_att2, indices2 = cnn(data)
                means2, log_var2 = netE1(input_att2)
                # means_record_sum += torch.sum(means, dim=0, keepdim=True)
                means2_record_sum_list.append(torch.sum(means2, dim=0, keepdim=True))
                means2_record = torch.sum(torch.vstack(means2_record_sum_list), dim=0, keepdim=True)
                means2_record = means2_record / train_size
                log_var2_record_sum_list.append(torch.sum(log_var2, dim=0, keepdim=True))
                log_var2_record = torch.sum(torch.vstack(log_var2_record_sum_list), dim=0, keepdim=True)
                log_var2_record = log_var2_record / train_size
                # std2 = torch.exp(0.5 * log_var2_record)
                std2 = log_var2_record
                input_res2 = netG(att=input_att2, mot=input_mot)
                eps2 = netE2(input_res2)
                z2 = means2_record.repeat(data.size(0), 1) + eps2 * std2
                recon_data12 = z2.unsqueeze(2)
                recon_data22 = recon_data12.expand(recon_data12.size(0), recon_data12.size(1), 49)
                recon_data32 = recon_data22.reshape(recon_data12.size(0), recon_data12.size(1), 7, 7)
                recon_data2= cnn_deconv(recon_data32, indices)
                loss5 = mse_loss_fig_fig(recon_data2, data)
                loss5_float.append(util.loss_to_float(loss5))
                loss5.backward()
                optimizerCnnDeconv.step()

                # if epoch == 0 and i==1:
                if (epoch+1)%args.save_epoch == 0 and i==1:
                    for image_i in range(5):
                        # deconv_img = a[image_i]
                        # ori_img = data[image_i]
                        for channel_i in range(3):
                            recon_data2[image_i][channel_i, :, :] = recon_data2[image_i][channel_i, :, :] * channel_std[channel_i] + channel_mean[channel_i]
                            data[image_i][channel_i, :, :] = data[image_i][channel_i, :, :] * channel_std[channel_i] + channel_mean[channel_i]
                        deconv_img = ToPILImage()(recon_data2[image_i])
                        img_name = all_loader.dataset.file_paths[index[image_i].detach().cpu().item()]
                        img_name = img_name.split('.')[0].split('/')[-1]
                        img_path_fake = os.path.join(log_dir, 'data_vis', img_name + '_' + str(epoch+1) + '_2.png')
                        ori_img = ToPILImage()(data[image_i])
                        deconv_img.save(img_path_fake)
                        if epoch == 0:
                            img_path_ori = os.path.join(log_dir, 'data_vis', img_name + '_1.png')
                            ori_img.save(img_path_ori)

        if (epoch + 1) % args.save_epoch == 0:
            tmp1 = (means2_record).t()
            tmp2 = tmp1.expand(512, 49)
            tmp3 = tmp2.unsqueeze(0)
            tmp4 = tmp3.reshape(1, 512, 7, 7)
            conv_indices = pool_indices[0].cpu().detach().sum(0, keepdim=True)
            for conv_in_i in range(1, len(pool_indices)):
                conv_indices += pool_indices[conv_in_i].cpu().detach().sum(0, keepdim=True)
            conv_indices = (conv_indices/train_size).ceil().long().to(device)
            deconv_img = cnn_deconv(tmp4, conv_indices).squeeze(0)
            for channel_i in range(3):
                deconv_img[channel_i, :, :] = deconv_img[channel_i, :, :] * channel_std[channel_i] + channel_mean[channel_i]
            img_path = os.path.join(log_dir, 'mean_vis', str(epoch+1)+'.png')
            deconv_img = ToPILImage()(deconv_img)
            deconv_img.save(img_path)

        # print('[%d/%d] loss1: %.4f, loss2: %.4f, loss3: %.4f, loss4:%.4f, loss5:%.4f, loss:%.4f, ' %
        #       (epoch+1, args.nepoch, loss1.item(), loss2.item(), loss3.item(),
        #        loss4.item(), loss5.item(), loss.item()))
        # print('[%d/%d] loss1: %.4f, loss2: %.4f, loss3: %.4f, loss4:%.4f, loss6:%.4f, loss_246:%.4f, loss5:%.4f' %
        #       (epoch + 1, args.nepoch, loss1.item(), loss2.item(), loss3.item(),
        #        loss4.item(), loss6.item(), loss_246.item(), loss5.item()))
        print('[%d/%d] loss2: %.4f, loss4:%.4f, loss6:%.4f, loss_246:%.4f, loss5:%.4f' %
              (epoch + 1, args.nepoch, np.array(loss2_float).mean(), np.array(loss4_float).mean(),
               np.array(loss6_float).mean(), np.array(loss_246_float).mean(), np.array(loss5_float).mean()))

        # writer.add_scalar("loss1", np.array(loss1_float).mean(), epoch + 1)
        writer.add_scalar("loss2", np.array(loss2_float).mean(), epoch + 1)
        # writer.add_scalar("loss3", np.array(loss3_float).mean(), epoch + 1)
        writer.add_scalar("loss4", np.array(loss4_float).mean(), epoch + 1)
        writer.add_scalar("loss6", np.array(loss6_float).mean(), epoch + 1)
        writer.add_scalar("loss_246", np.array(loss_246_float).mean(), epoch + 1)
        writer.add_scalar("loss5", np.array(loss5_float).mean(), epoch + 1)

        if (epoch+1)%args.save_epoch == 0:
            states = {}
            states['netE1'] = netE1.state_dict()
            states['netE2'] = netE2.state_dict()
            states['netG'] = netG.state_dict()
            states['netG_mot'] = netG_mot.state_dict()
            states['cnn'] = cnn.state_dict()
            states['cnn_deconv'] = cnn_deconv.state_dict()
            states['means'] = (means2_record).detach()
            states['log_var'] = (log_var2_record).detach()
            torch.save(states, os.path.join(log_dir, 'net_state', 'epoch'+str(epoch+1)))
            print('params of epoch %d are saved' % (epoch+1))

