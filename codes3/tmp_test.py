import argparse
import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as mp, seaborn
import matplotlib.pyplot as plt
import os.path as osp
from PIL import Image
from torchvision.transforms import ToPILImage

from materials import neutral_image, image_feature
from models import model, dis_model
from models.resnet import make_resnet18_base
from models.resnetDeconv import make_resnet18_deconv_base

import scipy.stats as stats
import scipy.optimize as opt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='cuda:1', help='gpu choice')
    parser.add_argument('--log_dir', type=str, default="save/try0102_1", help='log_dir')
    parser.add_argument('--means_file', type=str,
                        default='save/try0102_1/net_state/epoch200',
                        help='net state file')
    parser.add_argument('--mot_file', type=str,
                        default='save/try0102_1/mot_gen_2/mot_gen_epoch100',
                        help='net state file')
    parser.add_argument('--size_mot', type=int, default=512, help='dimension of motion')
    parser.add_argument('--size_att', type=int, default=512, help='dimension of attribute')
    parser.add_argument('--size_res', type=int, default=512, help='dimension of image feature')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[512, 1024])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 512])
    parser.add_argument('--GRAY', default=True, help='generate gray image or not')

    parser.add_argument("--encoder_use_mot", default=False, help="Encoder use motion as input")

    return parser.parse_args()


def cal_cos_similarity(imgs, att, mot, res):
    cos_att = torch.zeros(imgs.size(0), imgs.size(0))
    cos_mot = torch.zeros(imgs.size(0), imgs.size(0))
    cos_res = torch.zeros(imgs.size(0), imgs.size(0))
    for i in range(imgs.size(0)):
        for j in range(i + 1, imgs.size(0)):
            cos_att[i][j] = torch.cosine_similarity(att[i].reshape(-1, 512), att[j].reshape(-1, 512))
            cos_mot[i][j] = torch.cosine_similarity(mot[i].reshape(-1, 512), mot[j].reshape(-1, 512))
            cos_res[i][j] = torch.cosine_similarity(res[i].reshape(-1, 512), res[j].reshape(-1, 512))
    cos_att = cos_att.cpu().detach().numpy()
    cos_mot = cos_mot.cpu().detach().numpy()
    cos_res = cos_res.cpu().detach().numpy()

    seaborn.heatmap(cos_att, center=0, annot=False, xticklabels=list(range(imgs.size(0))),
                    yticklabels=list(range(imgs.size(0))))
    mp.title('cos_att')
    mp.show()
    seaborn.heatmap(cos_mot, center=0, annot=False, xticklabels=list(range(imgs.size(0))),
                    yticklabels=list(range(imgs.size(0))))
    mp.title('cos_mot')
    mp.show()
    seaborn.heatmap(cos_res, center=0, annot=False, xticklabels=list(range(imgs.size(0))),
                    yticklabels=list(range(imgs.size(0))))
    mp.title('cos_res')
    mp.show()
    seaborn.heatmap(cos_mot - cos_res, center=0, annot=False, xticklabels=list(range(imgs.size(0))),
                    yticklabels=list(range(imgs.size(0))))
    mp.title('cos_mot-cos_res')
    mp.show()

    return cos_att, cos_mot, cos_res


def cal_test_vis(imgs, att):
    channel_mean = [0.485, 0.456, 0.406]
    channel_std = [0.229, 0.224, 0.225]
    recon_tmp1 = att.unsqueeze(2)
    recon_tmp2 = recon_tmp1.expand(recon_tmp1.size(0), recon_tmp1.size(1), 49)
    recon_tmp3 = recon_tmp2.reshape(recon_tmp2.size(0), recon_tmp2.size(1), 7, 7)
    recon_data = cnn_deconv(recon_tmp3)
    for image_i in range(int(imgs.size(0) / 2)):
        # deconv_img = a[image_i]
        # ori_img = data[image_i]
        for channel_i in range(1):
            recon_data[image_i][channel_i, :, :] = recon_data[image_i][channel_i, :, :] * channel_std[
                channel_i] + channel_mean[channel_i]
        att_img = ToPILImage()(recon_data[image_i])
        vis_name = subloader.dataset.file_paths[indexes[image_i].detach().cpu().item()]
        vis_name = vis_name.split('.')[0].split('/')[-1]
        att_path = os.path.join(args.log_dir, 'test_vis', vis_name + '_att_eps.png')
        att_img.save(att_path)
    recon_tmp1 = means.unsqueeze(2)
    recon_tmp2 = recon_tmp1.expand(recon_tmp1.size(0), recon_tmp1.size(1), 49)
    recon_tmp3 = recon_tmp2.reshape(recon_tmp2.size(0), recon_tmp2.size(1), 7, 7)
    recon_means = cnn_deconv(recon_tmp3)
    img_path = os.path.join(args.log_dir, 'test_vis', 'means.png')
    means_img = ToPILImage()(recon_means[0])
    means_img.save(img_path)

if __name__ == '__main__':
    args = parse_args()
    global device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    shutil.rmtree(os.path.join(args.log_dir, 'test_vis'))
    os.makedirs(os.path.join(args.log_dir, 'test_vis'))

    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()

    means_f = torch.load(args.means_file)
    means = (means_f['means']).to(device)
    std = (means_f['std']).to(device)

    cnn_deconv = make_resnet18_deconv_base(args.GRAY).to(device)
    sd = cnn_deconv.state_dict()
    sd.update(means_f['cnn_deconv'])
    cnn_deconv.load_state_dict(sd)

    mot_f = torch.load(args.mot_file)
    cnn = make_resnet18_base().to(device)
    # pre_state_dict = torch.load('materials/resnet18-base.pth')
    # new_state_dict = {}
    # for k, v in cnn.state_dict().items():
    #     if k in pre_state_dict.keys() and k != 'conv1.weight':
    #         new_state_dict[k] = pre_state_dict[k]  # 如果原模型的层也在新模型的层里面， 那新模型就加载原先训练好的权重
    # cnn.load_state_dict(new_state_dict, False)
    sd = cnn.state_dict()
    sd.update(mot_f['cnn'])
    cnn.load_state_dict(sd)

    cls_logits = dis_model.Discriminator_D1(args, 6).to(device)
    sd = cls_logits.state_dict()
    sd.update(mot_f['Dis_mot'])
    cls_logits.load_state_dict(sd)

    netE2 = dis_model.Encoder2(args).to(device)
    sd = netE2.state_dict()
    sd.update(mot_f['netE2'])
    netE2.load_state_dict(sd)

    netG_mot = dis_model.Generator_mot(args)
    sd = netG_mot.state_dict()
    sd.update(mot_f['netG_mot'])
    netG_mot.load_state_dict(sd)
    netG_mot.to(device)

    train_loader, test_loader = image_feature.getRAFdata()
    test_nodes = train_loader.dataset.nodes

    netE2.eval()
    netG_mot.eval()
    cnn.eval()

    res_accT = []
    mot_accT = []
    flag = 1

    with torch.no_grad():
        for label_i, label in enumerate(test_nodes):
            subloader = test_loader[label_i]
            # subloader = train_loader
            for batch_i, (imgs, targets, indexes) in enumerate(subloader):

                imgs = imgs.to(device)
                res, indices = cnn(imgs)
                eps = netE2(res)
                att = means.repeat(imgs.size(0), 1) + eps * std
                mot = netG_mot(res, att)

                if flag == 1:
                    cos_att, cos_mot, cos_res = cal_cos_similarity(imgs, att, mot, res)
                    cal_test_vis(imgs, att)
                    flag = 0

                res_logits = cls_logits(res)
                mot_logits = cls_logits(mot)
                _, res_pred = torch.max(res_logits, dim=1)
                res_accT += torch.eq(res_pred, targets.to(device)).type(torch.FloatTensor)
                _, att_pred = torch.max(mot_logits, dim=1)
                mot_accT += torch.eq(att_pred, targets.to(device)).type(torch.FloatTensor)

        print('acc of res: %.4f, acc of mot: %.4f' % (np.array(res_accT).mean(), np.array(mot_accT).mean()))

            #     break
            # break