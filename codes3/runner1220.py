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
from models import dis_model, model, incep
from models.resnet import make_resnet18_base
from models.resnetDeconv import make_resnet18_deconv_base


def train(device, hyper_dict):
    args = hyper_dict
    channel_mean = [0.485, 0.456, 0.406]
    channel_std = [0.229, 0.224, 0.225]

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
    writer = SummaryWriter(log_dir)
    if args.continue_from != 0:
        f = torch.load(args.continue_file)
        cnn.load_state_dict(f['cnn'])
        cnn_deconv.load_state_dict(f['cnn_deconv'])
        netE1.load_state_dict(f['netE1'])
        netE2.load_state_dict(f['netE2'])
        netG.load_state_dict(f['netG'])
        netG_mot.load_state_dict(f['netG_mot'])
        means = f['means'].to(device)
        log_var = f['log_var'].to(device)
    else:
        folders = ['data_vis', 'mean_vis', 'test_vis', 'net_state']
        for folder in folders:
            if os.path.exists(os.path.join(log_dir, folder)):
                shutil.rmtree(os.path.join(log_dir, folder))
                os.makedirs(os.path.join(log_dir, folder))
            else:
                os.makedirs(os.path.join(log_dir, folder))

    optimG_mot = optim.Adam(netG_mot.parameters(), lr=args.lr)#, betas=(args.beta1, 0.999))
    optimCnnDeconv = optim.Adam(cnn_deconv.parameters(), lr=args.lr * 10.0)#, betas=(args.beta1, 0.999))
    optimAtt = optim.Adam(
        [{'params': cnn.parameters(), 'lr': args.lr},
         {'params': netE1.parameters(), 'lr': args.lr},
         {'params': netG.parameters(), 'lr': args.lr},
         {'params': netE2.parameters(), 'lr': args.lr}
         ])

    train_loader, test_loader, all_loader = neutral_image.getRAFdata()

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

        lossT_z = []
        lossT_mot = []
        lossT_deconv = []

        for step, (data, mot, index) in enumerate(all_loader, 1):
            torch.cuda.empty_cache()
            train_size = train_size + data.size(0)
            data = data.to(device)
            input_att, indices = cnn(data)  # resnet18生成的图片特征被认为是人脸属性特征
            input_mot = mot.to(device)  # noise被认为是中性表情的动作特征
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

            means_record = means.data
            log_var_record = log_var.data
            std_record = std.data

            eps = netE2(input_res)
            z = means.repeat(data.size(0), 1) + eps * std
            fake_mot = netG_mot(input_res, z)
            loss_mot = MSE(input_mot, fake_mot)
            lossT_mot.append(util.loss_to_float(loss_mot))
            loss_z_r = MSE(input_att, z)

            fake_means_t, fake_log_var_t = netE1(z)
            fake_means = Variable(torch.mean(fake_means_t, dim=0, keepdim=True), requires_grad=False)
            fake_log_var = Variable(torch.mean(fake_log_var_t, dim=0, keepdim=True), requires_grad=False)
            fake_std = torch.exp(0.5 * fake_log_var)
            fake_res = netG(att=z, mot=fake_mot)
            fake_eps = netE2(fake_res)
            fake_z = fake_means.repeat(data.size(0), 1) + fake_eps * fake_std
            loss_z_f = MSE(z, fake_z)
            loss_z = loss_z_r + loss_z_f
            lossT_z.append(util.loss_to_float(loss_z))
            # loss = loss_z + loss_mot

            optimAtt.zero_grad()
            optimG_mot.zero_grad()
            loss_z.backward(retain_graph=True)
            loss_mot.backward()
            # loss.backward()
            optimAtt.step()
            optimG_mot.step()

            if step % args.Deconv_step == 0:
                optimCnnDeconv.zero_grad()

                cnn.eval()
                cnn_deconv.train()
                netG.eval()
                netG_mot.eval()
                netE1.eval()
                netE2.eval()

                eps = netE2(input_res)
                z = means_record.repeat(data.size(0), 1) + eps.data * std_record
                recon_tmp1 = z.unsqueeze(2)
                recon_tmp2 = recon_tmp1.expand(recon_tmp1.size(0), recon_tmp1.size(1), 49)
                recon_tmp3 = recon_tmp2.reshape(recon_tmp2.size(0), recon_tmp2.size(1), 7, 7)
                recon_data = cnn_deconv(recon_tmp3)
                loss_deconv = MSE(recon_data, data)
                lossT_deconv.append(util.loss_to_float(loss_deconv))
                loss_deconv.backward()
                optimCnnDeconv.step()

        writer.add_scalar("lossT_mot", np.array(lossT_mot).mean(), epoch + 1)
        writer.add_scalar("lossT_z", np.array(lossT_z).mean(), epoch + 1)
        writer.add_scalar("lossT_deconv", np.array(lossT_deconv).mean(), epoch + 1)
        print('[%d/%d] loss of mot generation: %.4f, loss_z (total): %.4f, loss_deconv: %.4f' %
              (epoch + 1, args.nepoch, np.array(lossT_mot).mean(), np.array(lossT_z).mean(),
               np.array(lossT_deconv).mean()))

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
            torch.save(states, os.path.join(log_dir, 'net_state', 'epoch' + str(epoch + 1)))
            print('params of epoch %d are saved' % (epoch + 1))




