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

from materials import neutral_image, image_feature
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
    parser.add_argument('--dataset', type=str, default="RAF")
    parser.add_argument('--log_dir', type=str, default="save/RAF/try0118", help='log_dir')
    parser.add_argument('--means_file', type=str,
                        default='save/RAF/try0118/net_state/epoch200',
                        help='net state file')
    parser.add_argument('--mot_file', type=str,
                        default='save/RAF/try0118/mot_gen2/mot_gen_epoch120',
                        help='net state file')
    parser.add_argument('--size_mot', type=int, default=512, help='dimsension of motion')
    parser.add_argument('--size_att', type=int, default=512, help='dimension of attribute')
    parser.add_argument('--size_res', type=int, default=512, help='dimension of image feature')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[512, 1024])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 512])
    parser.add_argument('--GRAY', default=True, help='generate gray image or not')

    parser.add_argument("--encoder_use_mot", default=False, help="Encoder use motion as input")

    return parser.parse_args()


def plot_embedding(data, label, title):
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    # 创建了一个figure，标题为"Manifold Learning with 1000 points, 10 neighbors"
    plt.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    for i in range(data.shape[0]):
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=plt.cm.Set3(label[i]), cmap=plt.cm.Spectral)
    ax.view_init(4, -72)

    # fig = plt.figure()
    # ax = plt.subplot(111)
    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(label[i]),
    #              color=plt.cm.Set3(label[i] / 10.),
    #              fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(title)
    plt.show()
    # return fig


def tSNE_show(feats, labels, title):
    feat = feats.cpu().detach().numpy()
    label = labels.cpu().detach().numpy().ravel().tolist()
    print('Computing t-SNE embedding of %s' % title)
    tsne = TSNE(n_components=3, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(feat[:500])
    fig = plot_embedding(result, label, title)
    # plt.show(fig)


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
        train_loader, test_loader = image_feature.getRAFdata()
    elif args.dataset == 'AffectNet':
        train_loader, test_loader = image_feature.getAffectdata()
    elif args.dataset == 'FERPLUS':
        train_loader, test_loader = image_feature.getFERdata()
    test_nodes = train_loader.dataset.nodes
    num_cls = len(test_nodes)

    cls_logits = dis_model.Discriminator_mot(args, cls_num=num_cls).to(device)
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

    netE2.eval()
    netG_mot.eval()
    cnn.eval()

    res_accT = []
    mot_accT = []
    mot_feats = []
    res_feats = []
    labels = []
    flag = 1

    with torch.no_grad():
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            if imgs is not None:
                imgs = imgs.to(device)
                res, indices = cnn(imgs)
                eps = netE2(res)
                att = means.repeat(imgs.size(0), 1) + eps * std
                mot = netG_mot(res, att)
                mot_feats.append(mot)
                res_feats.append(res)
                labels.append(targets.reshape(-1, 1))
        mot_feats = torch.vstack(mot_feats)
        res_feats = torch.vstack(res_feats)
        labels = torch.vstack(labels)
        tSNE_show(mot_feats, labels, 'mot')
        tSNE_show(res_feats, labels, 'res')
            #     break
            # break