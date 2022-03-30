import os
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision.transforms import ToPILImage
import numpy as np

from materials import neutral_image
import util
from models import dis_model, model
from models.resnet import make_resnet18_base
from models.resnetDeconv import make_resnet18_deconv_base

import argparse
import random
import datetime
import pytz


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(device, hyper_dict):
    args = hyper_dict
    channel_mean = [0.485, 0.456, 0.406]
    channel_std = [0.229, 0.224, 0.225]

    if args.dataset == 'RAF':
        train_loader, test_loader, all_loader = neutral_image.getRAFdata()
        args.nepoch = 200
        args.save_epoch = 20
    elif args.dataset == 'AffectNet':
        train_loader, test_loader, all_loader = neutral_image.getAffectdata()
        args.nepoch = 100
        args.save_epoch = 10
    elif args.dataset == 'FERPLUS':
        train_loader, test_loader, all_loader = neutral_image.getFERdata()
        args.nepoch = 200
        args.save_epoch = 20
    elif args.dataset == 'CK+':
        train_loader, test_loader, all_loader = neutral_image.getCKdata()
        args.nepoch = 100
        args.save_epoch = 10
        # args.lr = args.lr / 10.0
    elif args.dataset == 'CASME2':
        train_loader, test_loader, all_loader = neutral_image.getCASME2data()
        args.nepoch = 100
        args.save_epoch = 10
    elif args.dataset == 'SAMM':
        train_loader, test_loader, all_loader = neutral_image.getSAMMdata()
        args.nepoch = 100
        args.save_epoch = 10

    cnn = make_resnet18_base().to(device)
    if args.GRAY is True:
        pre_state_dict = torch.load('materials/resnet18-base.pth')
        new_state_dict = {}
        for k, v in cnn.state_dict().items():
            if k in pre_state_dict.keys() and k != 'conv1.weight':
                new_state_dict[k] = pre_state_dict[k]  # 如果原模型的层也在新模型的层里面， 那新模型就加载原先训练好的权重
        cnn.load_state_dict(new_state_dict, False)
    else:
        sd = cnn.state_dict()
        sd.update(torch.load('materials/resnet18-base.pth'))
        cnn.load_state_dict(sd)

    cnn_deconv = make_resnet18_deconv_base(args.GRAY).to(device)

    netE1 = dis_model.Encoder1(args).to(device)
    netE2 = dis_model.Encoder2(args).to(device)  # 编码器
    netG = dis_model.Generator(args).to(device)  # 生成器
    netG_mot = dis_model.Generator_mot(args).to(device)

    log_dir = args.log_dir
    if args.continue_from != 0:
        f = torch.load(args.continue_file)
        cnn.load_state_dict(f['cnn'])
        cnn_deconv.load_state_dict(f['cnn_deconv'])
        netE1.load_state_dict(f['netE1'])
        netE2.load_state_dict(f['netE2'])
        netG.load_state_dict(f['netG'])
        netG_mot.load_state_dict(f['netG_mot'])
        means = f['means'].to(device)
        std = f['std'].to(device)
    else:
        folders = ['data_vis', 'mean_vis', 'test_vis', 'net_state']
        for folder in folders:
            if os.path.exists(os.path.join(log_dir, folder)):
                shutil.rmtree(os.path.join(log_dir, folder))
                os.makedirs(os.path.join(log_dir, folder))
            else:
                os.makedirs(os.path.join(log_dir, folder))
    writer = SummaryWriter(os.path.join(log_dir, 'mean_vis'))

    # optimG_mot = optim.Adam(netG_mot.parameters(), lr=args.lr)#, betas=(args.beta1, 0.999))
    optimCnnDeconv = optim.Adam(cnn_deconv.parameters(), lr=args.lr)#, betas=(0.9, 0.999))
    deconv_scheduler = optim.lr_scheduler.ExponentialLR(optimCnnDeconv, 0.9)
    # optimCnnDeconv = optim.SGD(cnn_deconv.parameters(), lr=args.lr * 2.0, momentum = 0.9)
    optimAtt = optim.SGD(
        [{'params': cnn.parameters()},
         {'params': netE1.parameters()},
         {'params': netG.parameters()},
         {'params': netE2.parameters()},
         {'params': netG_mot.parameters()},
         ], lr = args.lr, momentum = 0.9, weight_decay=1e-4
        # , betas=(0.9, 0.999), weight_decay=1e-4
    )
    Att_scheduler = optim.lr_scheduler.ExponentialLR(optimAtt, 0.9)

    MSE = nn.MSELoss()

    for epoch in range(args.continue_from, args.nepoch):

        torch.cuda.empty_cache()
        train_size = 0

        cnn.train()
        cnn_deconv.train()
        netG.train()
        netG_mot.train()
        netE1.train()
        netE2.train()

        lossT_z_r = []
        lossT_z_f = []
        lossT_z = []
        lossT_mot = []
        lossT= []
        lossT_deconv = []

        for step, (data, mot, index) in enumerate(train_loader, 1):
            optimAtt.zero_grad()

            torch.cuda.empty_cache()
            train_size = train_size + data.size(0)
            data = data.to(device)
            input_att, indices = cnn(data)  # resnet18生成的图片特征被认为是人脸属性特征
            input_mot = torch.squeeze(mot, 1).to(device)  # noise被认为是中性表情的动作特征
            input_res = netG(att=input_att, mot=input_mot)

            if step == 1:
                means_t, log_var_t = netE1(input_att)
                means = Variable(torch.mean(means_t, dim=0, keepdim=True), requires_grad=True)
                log_var = Variable(torch.mean(log_var_t, dim=0, keepdim=True), requires_grad=True)
                std = torch.exp(0.5 * log_var)
            else:
                means = Variable(means_record, requires_grad=True)
                log_var = Variable(log_var_record, requires_grad=True)
                std = torch.exp(0.5 * log_var)

            eps = netE2(input_res)
            z = means.repeat(data.size(0), 1) + eps * std
            fake_mot = netG_mot(input_res, z)
            loss_mot = MSE(input_mot, fake_mot)
            lossT_mot.append(util.loss_to_float(loss_mot))
            loss_z_r = MSE(input_att, z)

            # fake_means_t, fake_log_var_t = netE1(z)
            fake_means_t, fake_log_var_t = netE1(input_att)
            fake_means = Variable(torch.mean(fake_means_t, dim=0, keepdim=True), requires_grad=False)
            fake_log_var = Variable(torch.mean(fake_log_var_t, dim=0, keepdim=True), requires_grad=False)
            fake_std = torch.exp(0.5 * fake_log_var)
            fake_res = netG(att=z, mot=fake_mot)
            fake_eps = netE2(fake_res)
            fake_z = fake_means.repeat(data.size(0), 1) + fake_eps * fake_std
            loss_z_f = MSE(input_att, fake_z)

            lossT_z_r.append(util.loss_to_float(loss_z_r))
            lossT_z_f.append(util.loss_to_float(loss_z_f))
            loss_z = loss_z_r + loss_z_f
            lossT_z.append(util.loss_to_float(loss_z))
            loss = loss_z + loss_mot
            lossT.append(util.loss_to_float(loss))

            # optimG_mot.zero_grad()
            # loss_z.backward(retain_graph=True)
            # loss_mot.backward()
            loss.backward()
            optimAtt.step()
            # Att_scheduler.step()
            # optimG_mot.step()
            means_record = means.data
            log_var_record = log_var.data
            std_record = torch.exp(0.5 * log_var.data)

            if step % args.Deconv_step == 0:
                optimCnnDeconv.zero_grad()

                cnn.eval()
                cnn_deconv.train()
                netG.eval()
                netG_mot.eval()
                netE1.eval()
                netE2.eval()

                z = means_record.repeat(data.size(0), 1) + eps.data * std_record
                recon_tmp1 = z.unsqueeze(2)
                recon_tmp2 = recon_tmp1.expand(recon_tmp1.size(0), recon_tmp1.size(1), 49)
                recon_tmp3 = recon_tmp2.reshape(recon_tmp2.size(0), recon_tmp2.size(1), 7, 7)
                recon_data = cnn_deconv(recon_tmp3)
                loss_deconv = MSE(recon_data, data)
                lossT_deconv.append(util.loss_to_float(loss_deconv))

                loss_deconv.backward()
                optimCnnDeconv.step()
                # deconv_scheduler.step()

        writer.add_scalar("lossT_mot", np.array(lossT_mot).mean(), epoch + 1)
        writer.add_scalar("lossT_z", np.array(lossT_z).mean(), epoch + 1)
        writer.add_scalar("loss_z_r", np.array(lossT_z_r).mean(), epoch + 1)
        writer.add_scalar("loss_z_f", np.array(lossT_z_f).mean(), epoch + 1)
        writer.add_scalar("lossT", np.array(lossT).mean(), epoch + 1)
        writer.add_scalar("lossT_deconv", np.array(lossT_deconv).mean(), epoch + 1)

        print('[%d/%d] lossT_z: %.4f, lossT_mot: %.4f, lossT: %.4f, loss_deconv: %.4f' %
              (epoch + 1, args.nepoch, np.array(lossT_z).mean(), np.array(lossT_mot).mean(),
               np.array(lossT).mean(), np.array(lossT_deconv).mean()))

        if (epoch+1) % args.save_epoch == 0:
            torch.cuda.empty_cache()
            for image_i in range(2):
                img_name = str(int(index[image_i]))
                img_path_ori = os.path.join(log_dir, 'data_vis', img_name + '_' + str(epoch + 1) + '_1.png')
                data[image_i][0, :, :] = data[image_i][0, :, :] * channel_std[0] + channel_mean[0]
                ori_img = ToPILImage()(data[image_i])
                ori_img.save(img_path_ori)
                img_path_fake = os.path.join(log_dir, 'data_vis', img_name + '_' + str(epoch + 1) + '_2.png')
                recon_data[image_i][0, :, :] = recon_data[image_i][0, :, :] * channel_std[0] + channel_mean[0]
                deconv_img = ToPILImage()(recon_data[image_i])
                deconv_img.save(img_path_fake)
            recon_tmp1 = means_record.unsqueeze(2)
            recon_tmp2 = recon_tmp1.expand(recon_tmp1.size(0), recon_tmp1.size(1), 49)
            recon_tmp3 = recon_tmp2.reshape(recon_tmp2.size(0), recon_tmp2.size(1), 7, 7)
            recon_means = cnn_deconv(recon_tmp3)
            img_path = os.path.join(log_dir, 'mean_vis', str(epoch + 1) + '.png')
            means_img = ToPILImage()(recon_means[0])
            means_img.save(img_path)

            states = {}
            states['netE1'] = netE1.state_dict()
            states['netE2'] = netE2.state_dict()
            states['netG'] = netG.state_dict()
            states['netG_mot'] = netG_mot.state_dict()
            states['cnn'] = cnn.state_dict()
            states['cnn_deconv'] = cnn_deconv.state_dict()
            states['means'] = means_record.detach()
            states['std'] = std.detach()
            states['lr'] = optimAtt.param_groups[0]['lr']
            torch.save(states, os.path.join(log_dir, 'net_state', 'epoch' + str(epoch + 1)))
            print('params of epoch %d are saved' % (epoch + 1))


def mean_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='cuda:1', help='gpu choice')
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
    parser.add_argument('--continue_from', type=int, default=0, help='save interval')
    parser.add_argument('--dataset', type=str, default="RAF")
    parser.add_argument('--log_dir', type=str, default="save", help='log_dir')
    parser.add_argument('--continue_file', type=str,
                        default=None,
                        help='net state file')
    parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=10, help='number of epochs to save')
    parser.add_argument('--Deconv_step', type=int, default=5, help='number of epochs to train Deconv')
    parser.add_argument('--GRAY', default=True, help='generate gray image or not')

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--size_mot', type=int, default=512, help='dimension of motion')
    parser.add_argument('--size_att', type=int, default=512, help='dimension of attribute')
    parser.add_argument('--size_res', type=int, default=512, help='dimension of image feature')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[512, 1024])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 512])

    parser.add_argument("--encoder_use_mot", default=False, help="Encoder use motion as input")

    return parser.parse_args()


if __name__ == '__main__':
    mean_args = mean_parse_args()
    torch.cuda.empty_cache()
    global device
    device = torch.device(mean_args.gpu if torch.cuda.is_available() else 'cpu')
    print('using gpu:', mean_args.gpu)

    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)
    cur_day = str(cur_time).split(' ')
    cur_day = cur_day[0]
    mean_args.log_dir = os.path.join(mean_args.log_dir, mean_args.dataset, cur_day, 'encoderSGDwAdam')

    print('---- START %s Gauss Encoder ----' % (mean_args.dataset))
    if mean_args.manualSeed is None:
        mean_args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", mean_args.manualSeed)
    setup_seed(mean_args.manualSeed)

    train(device, mean_args)
