import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision.transforms import ToPILImage
import numpy as np
import os.path as osp
import time
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
    parser.add_argument('--manualSeed', type=int, default=20, help='manual seed')
    parser.add_argument('--continue_from', type=int, default=0, help='save interval')
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
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

# gpu设置
global device
# device = torch.device('cpu')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.manualSeed)

train_loader, test_loader, all_loader = neutral_image.getRAFdata()

# f = torch.load('/media/database/data4/wf/IFER/codes2/mot_gaus/50deconv_states')
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

optimizerE1 = optim.Adam(netE1.parameters(), lr=args.lr)
optimizerE2 = optim.Adam(netE2.parameters(), lr=args.lr)
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG_mot = optim.Adam(netG_mot.parameters(), lr=0.001, betas=(args.beta1, 0.999))
optimizerCNN = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
optimizerCnnDeconv = optim.SGD(cnn_deconv.parameters(), lr=0.01,
                      momentum=0.9,
                      weight_decay=5e-4,
                      nesterov=True)

print('Current Training time', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
# print(datetime.datetime.now())

log_dir = "mot_gaus"
writer = SummaryWriter(log_dir)

channel_mean=[0.485, 0.456, 0.406]
channel_std=[0.229, 0.224, 0.225]

for epoch in range(args.continue_from, args.nepoch):
    torch.cuda.empty_cache()
    print(epoch, end=' ')
    if epoch < 500:
        cnn.train()
        cnn_deconv.train()
        netG.train()
        netG_mot.train()
        netE1.train()
        netE2.train()

        assert netG.training
        assert netG_mot.training
        assert netE1.training
        assert netE2.training
        assert cnn.training
        assert cnn_deconv.training
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
    loss_float = []

    means_record_sum = torch.zeros(1, args.size_att).to(device)
    log_var_record_sum = torch.zeros(1, args.size_att).to(device)
    fake_means_record_sum = torch.zeros(1, args.size_att).to(device)
    fake_log_var_record_sum = torch.zeros(1, args.size_att).to(device)

    pool_indices = []

    with torch.autograd.set_detect_anomaly(True):
        for i, (data, index) in enumerate(all_loader, 1):
            if data is None:
                flag_not += 1
                continue
            else:
                train_size = train_size + data.size(0)
                data = data.to(device)
                input_att, indices = cnn(data) # resnet18生成的图片特征被认为是人脸属性特征
                pool_indices.append(indices)
                noise = torch.randn(data.size(0), args.size_mot).to(device)  # 随机生成一个标准正态分布的noise
                input_mot = noise # noise被认为是中性表情的动作特征

                input_res = netG(att=input_att, mot=input_mot)
                loss1 = mse_loss_att_res(input_att, input_res)

                means, log_var = netE1(input_res)
                means_record_sum += torch.sum(means, dim=0, keepdim=True)
                means_record = torch.mean(means, dim=0)
                # means_record = means_record_sum/train_size
                log_var_record_sum += torch.sum(log_var, dim=0, keepdim=True)
                log_var_record = torch.mean(log_var, dim=0)
                # log_var_record = log_var_record_sum/train_size
                std = torch.exp(0.5 * log_var_record)
                eps = netE2(input_res)
                z = means_record.repeat(data.size(0), 1) + eps*std
                loss2 = mse_loss_att_att(z, input_att)

                fake_res = netG(z, input_mot)
                loss3 = mse_loss_res_res(input_res, fake_res)

                fake_means, fake_log_var = netE1(fake_res)
                fake_means_record_sum += torch.sum(fake_means, dim=0, keepdim=True)
                fake_means_record = torch.mean(fake_means, dim=0)
                fake_log_var_record_sum += torch.sum(fake_log_var, dim=0, keepdim=True)
                fake_log_var_record = torch.mean(fake_log_var, dim=0)
                fake_std = torch.exp(0.5 * fake_log_var_record)
                fake_eps = netE2(fake_res)
                fake_z = fake_means_record.repeat(data.size(0), 1) + fake_eps*fake_std
                loss4 = mse_loss_att_att(z, fake_z)

                optimizerG.zero_grad()
                optimizerG_mot.zero_grad()
                optimizerE1.zero_grad()
                optimizerE1.zero_grad()
                optimizerCNN.zero_grad()
                optimizerCnnDeconv.zero_grad()

                # tmp1 = means_record.reshape(1, -1).t()
                # tmp2 = tmp1.expand(512, 49)
                # tmp3 = tmp2.unsqueeze(0)
                # tmp4 = tmp3.reshape(1, 512, 7, 7)
                q1 = z.unsqueeze(2)
                q2 = q1.expand(q1.size(0), q1.size(1), 49)
                q3 = q2.reshape(q1.size(0), q1.size(1), 7, 7)

                a = cnn_deconv(q3, indices)

                a_copy = a
                data_copy = data
                for image_i in range(data.size(0)):
                    # deconv_img = a[image_i]
                    # ori_img = data[image_i]
                    for channel_i in range(3):
                        a_copy[image_i][channel_i, :, :] = a_copy[image_i][channel_i, :, :] * channel_std[channel_i] + \
                                                      channel_mean[channel_i]
                        data_copy[image_i][channel_i, :, :] = data_copy[image_i][channel_i, :, :] * channel_std[channel_i] + \
                                                         channel_mean[channel_i]
                loss5  = mse_loss_fig_fig(a, data)

                fake_mot = netG_mot(input_res, input_att)
                loss6 = mse_loss_mot_mot(fake_mot, input_mot)

                loss = loss1+loss2+loss3+loss4+loss5+loss6

                loss1_float.append(util.loss_to_float(loss1))
                loss2_float.append(util.loss_to_float(loss2))
                loss3_float.append(util.loss_to_float(loss3))
                loss4_float.append(util.loss_to_float(loss4))
                loss5_float.append(util.loss_to_float(loss5))
                loss6_float.append(util.loss_to_float(loss6))
                loss_float.append(util.loss_to_float(loss))
                loss.backward()
                optimizerE1.step()
                optimizerE2.step()
                optimizerG.step()
                optimizerG_mot.step()
                optimizerCNN.step()
                optimizerCnnDeconv.step()

                # if epoch == 1:
                if epoch == args.nepoch-1:
                    for image_i in range(data.size(0)):
                        deconv_img = a[image_i]
                        ori_img = data[image_i]
                        for channel_i in range(3):
                            a[image_i][channel_i, :, :] = a[image_i][channel_i, :, :] * channel_std[channel_i] + channel_mean[channel_i]
                            data[image_i][channel_i, :, :] = data[image_i][channel_i, :, :] * channel_std[channel_i] + channel_mean[channel_i]
                        deconv_img = ToPILImage()(a[image_i])
                        img_name = all_loader.dataset.file_paths[index[image_i].detach().cpu().item()]
                        img_name = img_name.split('.')[0].split('/')[-1]
                        img_path_ori = os.path.join(log_dir, 'each_image', img_name + '_1.png')
                        img_path_fake = os.path.join(log_dir, 'each_image', img_name + '_2.png')
                        ori_img = ToPILImage()(data[image_i])
                        ori_img.save(img_path_ori)
                        deconv_img.save(img_path_fake)

        tmp1 = (means_record_sum/train_size).t()
        tmp2 = tmp1.expand(512, 49)
        tmp3 = tmp2.unsqueeze(0)
        tmp4 = tmp3.reshape(1, 512, 7, 7)

        conv_indices = pool_indices[0].cpu().detach().sum(0, keepdim=True)
        for conv_in_i in range(1, len(pool_indices)):
            conv_indices += pool_indices[conv_in_i].cpu().detach().sum(0, keepdim=True)
        conv_indices = (conv_indices/train_size).ceil().long().to(device)

        deconv_image = cnn_deconv(tmp4, conv_indices)
        deconv_img = ToPILImage()(a[0])
        img_path = os.path.join(log_dir, 'conv_image', str(epoch+1)+'.png')
        deconv_img.save(img_path)

        # print('[%d/%d] loss1: %.4f, loss2: %.4f, loss3: %.4f, loss4:%.4f, loss5:%.4f, loss:%.4f, ' %
        #       (epoch+1, args.nepoch, loss1.item(), loss2.item(), loss3.item(),
        #        loss4.item(), loss5.item(), loss.item()))
        # print('[%d/%d] loss1: %.4f, loss2: %.4f, loss3: %.4f, loss4:%.4f, loss5:%.4f, loss6:%.4f, loss:%.4f' %
        #       (epoch + 1, args.nepoch, loss1.item(), loss2.item(), loss3.item(),
        #        loss4.item(), loss5.item(), loss6.item(), loss.item()))
        #
        # writer.add_scalar("loss1", np.array(loss1_float).mean(), epoch + 1)
        # writer.add_scalar("loss2", np.array(loss2_float).mean(), epoch + 1)
        # writer.add_scalar("loss3", np.array(loss3_float).mean(), epoch + 1)
        # writer.add_scalar("loss4", np.array(loss4_float).mean(), epoch + 1)
        # writer.add_scalar("loss5", np.array(loss5_float).mean(), epoch + 1)
        # writer.add_scalar("loss6", np.array(loss_float).mean(), epoch + 1)
        # writer.add_scalar("loss", np.array(loss_float).mean(), epoch + 1)
        #
        # if (epoch+1)%args.save_epoch == 0:
        #     states = {}
        #     states['netE1'] = netE1.state_dict()
        #     states['netE2'] = netE2.state_dict()
        #     states['netG'] = netG.state_dict()
        #     states['netG_mot'] = netG_mot.state_dict()
        #     states['cnn'] = cnn.state_dict()
        #     states['cnn_deconv'] = cnn_deconv.state_dict()
        #     states['means'] = means_record_sum/train_size
        #     states['log_var'] = log_var_record_sum/train_size
        #     torch.save(states, os.path.join(log_dir, str(epoch+1)+'deconv_states'))
        #     print('params of epoch %d are saved' % (epoch+1))

