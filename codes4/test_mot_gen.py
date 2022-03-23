import argparse
from time import time
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

from materials import neutral_image, image_feature_nN
from models import model, dis_model
from models.resnet import make_resnet18_base
from models.resnetDeconv import make_resnet18_deconv_base

from sklearn.manifold import TSNE
import warnings

import scipy.stats as stats
import scipy.optimize as opt


def parse_args():
    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore")  # 忽略警告
    parser.add_argument('--gpu', type=str, default='cuda:1', help='gpu choice')
    parser.add_argument('--dataset', type=str, default="FERPLUS")
    parser.add_argument('--log_dir', type=str, default="save/FERPLUS/2022-03-18/decoder0322", help='log_dir')
    parser.add_argument('--means_file', type=str,
                        default='save/FERPLUS/2022-03-18/encoder/net_state/epoch100',
                        help='net state file')
    parser.add_argument('--mot_file', type=str,
                        default='save/FERPLUS/2022-03-18/decoder0322/mot_gen_epoch60',
                        help='net state file')
    parser.add_argument('--size_mot', type=int, default=512, help='dimsension of motion')
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


def cal_test_vis(imgs, feat, kind):
    channel_mean = [0.485, 0.456, 0.406]
    channel_std = [0.229, 0.224, 0.225]
    recon_tmp1 = feat.unsqueeze(2)
    recon_tmp2 = recon_tmp1.expand(recon_tmp1.size(0), recon_tmp1.size(1), 49)
    recon_tmp3 = recon_tmp2.reshape(recon_tmp2.size(0), recon_tmp2.size(1), 7, 7)
    recon_data = cnn_deconv(recon_tmp3)
    for image_i in range(int(imgs.size(0) / 2)):
        # deconv_img = a[image_i]
        # ori_img = data[image_i]
        for channel_i in range(1):
            recon_data[image_i][channel_i, :, :] = recon_data[image_i][channel_i, :, :] * channel_std[
                channel_i] + channel_mean[channel_i]
            imgs[image_i][channel_i, :, :] = imgs[image_i][channel_i, :, :] * channel_std[
                channel_i] + channel_mean[channel_i]
        att_img = ToPILImage()(recon_data[image_i])
        ori_img = ToPILImage()(imgs[image_i])
        vis_name = subloader.dataset.file_paths[indexes[image_i].detach().cpu().item()]
        vis_name = vis_name.split('.')[0].split('/')[-1].split('_')
        vis_name = vis_name[0] + '_' + vis_name[1] + kind
        if kind == 'att':
            ori_name = vis_name[0] + '_' + vis_name[1] + 'ori'
            ori_img.save(os.path.join(args.log_dir, 'test_vis', ori_name + '.png'))
        vis_name = vis_name + kind
        att_path = os.path.join(args.log_dir, 'test_vis', vis_name + '.png')
        att_img.save(att_path)

    if kind == 'att':
        recon_tmp1 = means.unsqueeze(2)
        recon_tmp2 = recon_tmp1.expand(recon_tmp1.size(0), recon_tmp1.size(1), 49)
        recon_tmp3 = recon_tmp2.reshape(recon_tmp2.size(0), recon_tmp2.size(1), 7, 7)
        recon_means = cnn_deconv(recon_tmp3)
        img_path = os.path.join(args.log_dir, 'test_vis', 'means.png')
        means_img = ToPILImage()(recon_means[0])
        means_img.save(img_path)

        recon_tmp1 = std.unsqueeze(2)
        recon_tmp2 = recon_tmp1.expand(recon_tmp1.size(0), recon_tmp1.size(1), 49)
        recon_tmp3 = recon_tmp2.reshape(recon_tmp2.size(0), recon_tmp2.size(1), 7, 7)
        recon_std = cnn_deconv(recon_tmp3)
        img_path = os.path.join(args.log_dir, 'test_vis', 'std.png')
        std_img = ToPILImage()(recon_std[0])
        std_img.save(img_path)


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set3(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    # return fig


def tSNE_show(feats, labels, title):
    feat = feats.cpu().detach().numpy()
    label = labels.cpu().detach().numpy().ravel().tolist()
    print('Computing t-SNE embedding of %s' % title)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(feat)
    fig = plot_embedding(result, label, title)
    # plt.show(fig)


if __name__ == '__main__':
    args = parse_args()
    global device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    if os.path.exists(os.path.join(args.log_dir, 'test_vis')) is True:
        shutil.rmtree(os.path.join(args.log_dir, 'test_vis'))
    else:
        os.makedirs(os.path.join(args.log_dir, 'test_vis'))

    means_f = torch.load(args.means_file)
    means = (means_f['means']).to(device)
    std = (means_f['std']).to(device)

    cnn_deconv = make_resnet18_deconv_base(args.GRAY).to(device)
    sd = cnn_deconv.state_dict()
    sd.update(means_f['cnn_deconv'])
    cnn_deconv.load_state_dict(sd)

    print('mot file', args.mot_file.split('/')[-1])
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

    if args.dataset == 'RAF':
        train_loader, test_loader = image_feature_nN.getRAFdata()
    elif args.dataset == 'AffectNet':
        train_loader, test_loader = image_feature_nN.getAffectdata()
    elif args.dataset == 'FERPLUS':
        train_loader, test_loader = image_feature_nN.getFERdata()
    elif args.dataset == 'CK+':
        train_loader, test_loader = image_feature_nN.getCKdata()
    test_nodes = train_loader.dataset.nodes
    num_cls = len(test_nodes)

    cls_logits = dis_model.Discriminator_mot(args, cls_num=num_cls).to(device)
    sd = cls_logits.state_dict()
    sd.update(mot_f['Dis_mot'])
    cls_name = 'Dis_mot'
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

    netE2.eval()
    netG_mot.eval()
    cnn.eval()
    cls_logits.eval()

    res_accT = []
    mot_accT = []
    mot_feats = []
    res_feats = []
    labels = []
    flag = 1

    with torch.no_grad():
        class_acc = []
        for label_i, label in enumerate(test_nodes):
            subloader = test_loader[label_i]
            # subloader = train_loader
            for batch_i, (imgs, targets, indexes) in enumerate(subloader):
                if imgs is not None:
                    imgs = imgs.to(device)
                    res, indices = cnn(imgs)
                    eps = netE2(res)
                    att = means.repeat(imgs.size(0), 1) + eps * std
                    mot = netG_mot(res, att)
                    mot_feats.append(mot)
                    res_feats.append(res)
                    labels.append(targets.reshape(-1, 1))

                    # if label_i == 2 and batch_i == 1:
                    # if flag == 1:
                    #     cos_att, cos_mot, cos_res = cal_cos_similarity(imgs, att, mot, res)
                    #     cal_test_vis(imgs, att, 'att')
                    #     cal_test_vis(imgs, mot, 'mot')
                    #     flag = 0

                    # res_logits = cls_logits(res)
                    mot_logits = cls_logits(mot)
                    # _, res_pred = torch.max(res_logits, dim=1)
                    # res_accT += torch.eq(res_pred, targets.to(device)).type(torch.FloatTensor)
                    _, mot_pred = torch.max(mot_logits, dim=1)
                    mot_accT += torch.eq(mot_pred, targets.to(device)).type(torch.FloatTensor)
                    class_acc += torch.eq(mot_pred, targets.to(device)).type(torch.FloatTensor)

            print('%s acc is %.4f' % (test_nodes[label_i], np.array(class_acc).mean()))
            class_acc = []

        print('using %s, acc of dataset: %.4f' % (cls_name, np.array(mot_accT).mean()))
        mot_feats = torch.vstack(mot_feats)
        # res_feats = torch.vstack(res_feats)
        labels = torch.vstack(labels)
        # tSNE_show(mot_feats, labels, 'mot')
        # tSNE_show(res_feats, labels, 'res')