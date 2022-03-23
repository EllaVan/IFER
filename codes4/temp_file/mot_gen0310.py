import argparse
import random
import datetime
import pytz
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np

from materials import neutral_image, image_feature
import util
from models import model, dis_model
from models.resnet import make_resnet18_base


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mot_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu choice')
    parser.add_argument('--dataset', type=str, default="RAF")
    parser.add_argument('--means_file', type=str,
                        default='save/RAF/encoder/epoch200')
    parser.add_argument('--log_dir', type=str, default="save/RAF/decoder0315", help='log_dir')

    parser.add_argument('--start_epoch', type=int, default=0, help='save interval')
    parser.add_argument('--start_file', type=str,
                        default='save/RAF/2022-03-14/decoder0314/mot_gen_epoch20')
    parser.add_argument('--end_epoch', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=20, help='number of epochs to save for')
    parser.add_argument('--k_save_epoch', type=int, default=20, help='number of epochs to save for')


    parser.add_argument('--manualSeed', type=int, default=5012, help='manual seed')

    parser.add_argument('--num_query', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--size_mot', type=int, default=512, help='dimension of motion')
    parser.add_argument('--size_att', type=int, default=512, help='dimension of attribute')
    parser.add_argument('--size_res', type=int, default=512, help='dimension of image feature')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[512, 1024])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 512])

    parser.add_argument("--encoder_use_mot", default=False, help="Encoder use motion as input")

    return parser.parse_args()


def run(device, args):
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)
    print('using gpu:', str(device))
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    setup_seed(args.manualSeed)

    k = 1
    if args.dataset == 'RAF':
        train_loader, test_loader = image_feature.getRAFdata()
        flag_class = 6
        args.end_epoch = 400
        args.save_epoch = 40
    elif args.dataset == 'AffectNet':
        train_loader, test_loader = image_feature.getAffectdata()
        flag_class = 0
        args.end_epoch = 200
        args.save_epoch = 10
        # args.lr = 0.001
    elif args.dataset == 'FERPLUS':
        train_loader, test_loader = image_feature.getFERdata()
        flag_class = 0
        args.end_epoch = 400
        args.save_epoch = 40
        # args.lr = 0.001
    elif args.dataset == 'CK+':
        train_loader, test_loader = image_feature.getCKdata()
        flag_class = 6
        args.end_epoch = 200
        args.save_epoch = 10
        args.lr = args.lr / 10.0
        k = 5
        args.k_save_epoch = 50
    test_nodes = train_loader.dataset.nodes
    num_cls = len(test_nodes)

    if args.start_epoch == 0:
        args.start_file = args.means_file
        if os.path.exists(args.log_dir) is True:
            shutil.rmtree(args.log_dir)
            os.makedirs(args.log_dir)
        else:
            os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)

    CE = nn.CrossEntropyLoss()

    means_f = torch.load(args.means_file)
    means = (means_f['means']).to(device)
    std = (means_f['std']).to(device)

    mot_f = torch.load(args.start_file)
    cnn = make_resnet18_base().to(device)
    if args.start_epoch == 0:
        pre_state_dict = torch.load('materials/resnet18-base.pth')
        new_state_dict = {}
        for k, v in cnn.state_dict().items():
            if k in pre_state_dict.keys() and k != 'conv1.weight':
                new_state_dict[k] = pre_state_dict[k]  # 如果原模型的层也在新模型的层里面， 那新模型就加载原先训练好的权重
        cnn.load_state_dict(new_state_dict, False)
    else:
        sd = cnn.state_dict()
        sd.update(mot_f['cnn'])
        cnn.load_state_dict(sd)

    netE2 = dis_model.Encoder2(args).to(device)
    # sd = netE2.state_dict()
    # sd.update(mot_f['netE2'])
    # netE2.load_state_dict(sd)

    netG_mot = dis_model.Generator_mot(args).to(device)
    # sd = netG_mot.state_dict()
    # sd.update(mot_f['netG_mot'])
    # netG_mot.load_state_dict(sd)

    Dis_att = dis_model.Discriminator_att(args, cls_num=1).to(device)
    Dis_mot = dis_model.Discriminator_mot(args, cls_num=num_cls).to(device)
    if args.start_epoch != 0:
        sd = Dis_mot.state_dict()
        sd.update(mot_f['Dis_mot'])
        Dis_mot.load_state_dict(sd)

        sd = Dis_att.state_dict()
        sd.update(mot_f['Dis_att'])
        Dis_att.load_state_dict(sd)
    optim_mot = optim.Adam(
        [{'params': netG_mot.parameters(), 'lr': args.lr},
         {'params': Dis_mot.parameters(), 'lr': args.lr},
         {'params': Dis_att.parameters(), 'lr': args.lr/10.0},
         {'params': cnn.parameters(), 'lr': args.lr},
         {'params': netE2.parameters(), 'lr': args.lr}]
    )

    for epoch in range(1+args.start_epoch, 1+args.end_epoch):
        torch.cuda.empty_cache()
        netE2.train()
        netG_mot.train()
        Dis_att.train()
        Dis_mot.train()

        errT_mot = []
        errT_mot_att = []
        errT_mot_cls = []
        accT_mot_cls = []

        mot_record = []
        label_record = []

        for i, (data, label, index) in enumerate(train_loader, 1):

            torch.cuda.empty_cache()
            mot_mean = {}
            mot_feat = {}
            att_feat = {}
            for nodes_i in range(num_cls):
                mot_mean[nodes_i] = []
                mot_feat[nodes_i] = []
                att_feat[nodes_i] = []

            data = data.to(device)
            label = label.to(device)
            input_res, indices = cnn(data)  # resnet18生成的图片特征被认为是人脸属性特征
            eps = netE2(input_res)
            att = means.repeat(data.size(0), 1) + eps * std
            mot = netG_mot(input_res, att)
            mot_record.append(mot)
            label_record.append(label)

            mot_cls_logits = Dis_mot(mot)
            err_mot_cls = CE(mot_cls_logits, label)

            if epoch <= args.k_save_epoch:
                for label_i in range(data.size(0)):
                    mot_feat[label[label_i].item()].append(mot[label_i].data)
                    att_feat[label[label_i].item()].append(att[label_i].data)
                for class_i in range(num_cls):
                    if len(mot_feat[class_i]) != 0:
                        mot_mean[class_i] = torch.mean(torch.vstack(mot_feat[class_i]), dim=0, keepdim=True)
                    else:
                        mot_mean[class_i] = None
                K = 0
                err_mot_att_tmp = 0
                for class_i in range(num_cls):
                    if class_i != flag_class:
                        if len(mot_feat[class_i]) >= args.num_query:
                        # if len(mot_feat[class_i]) != 0:
                            for query_i in range(args.num_query):
                            # for query_i in range(len(mot_feat[class_i])):
                                err_mot_att_tmp += max(0, 1 - Dis_att(mot_feat[class_i][query_i].reshape(1, -1), mot_mean[class_i])) \
                                                   + max(0, 1 + Dis_att(att_feat[class_i][query_i].reshape(1, -1), mot_mean[class_i]))
                                K += 1
                    else:
                        if len(mot_feat[class_i]) != 0:
                            for query_i in range(len(mot_feat[class_i])):
                                err_mot_att_tmp += max(0, 1 - Dis_att(mot_feat[class_i][query_i].reshape(1, -1), mot_mean[class_i])) \
                                                   + max(0, Dis_att(att_feat[class_i][query_i].reshape(1, -1), mot_mean[class_i]))
                                K += 1
                if K != 0:
                    err_mot_att = err_mot_att_tmp / (K * 1.0)
                else:
                    err_mot_att = 0
                err_mot = err_mot_cls + err_mot_att
                errT_mot.append(util.loss_to_float(err_mot_cls) + util.loss_to_float(err_mot_att))
                errT_mot_att.append(util.loss_to_float(err_mot_att))
            else:
                err_mot = err_mot_cls
                errT_mot.append(util.loss_to_float(err_mot_cls))
            errT_mot_cls.append(util.loss_to_float(err_mot_cls))
            _, mot_cls_pred = torch.max(mot_cls_logits, dim=1)
            accT_mot_cls.append(torch.eq(mot_cls_pred, label).type(torch.FloatTensor).mean().item())

            optim_mot.zero_grad()
            err_mot.backward()
            optim_mot.step()

        writer.add_scalar("errT_mot", np.array(errT_mot).mean(), epoch)
        writer.add_scalar("errT_mot_cls", np.array(errT_mot_cls).mean(), epoch)
        if epoch <= args.k_save_epoch:
            writer.add_scalar("errT_mot_att", np.array(errT_mot_att).mean(), epoch)
            print(
                '[%d/%d] errT_mot: %.4f, errT_mot_att: %.4f, errT_mot_cls: %.4f, accT_mot_cls: %.4f' %
                (epoch, args.end_epoch, np.array(errT_mot).mean(), np.array(errT_mot_att).mean(),
                 np.array(errT_mot_cls).mean(), np.array(accT_mot_cls).mean()))
        else:
            print(
                '[%d/%d] errT_mot: %.4f, errT_mot_cls: %.4f, accT_mot_cls: %.4f' %
                (epoch, args.end_epoch, np.array(errT_mot).mean(),
                 np.array(errT_mot_cls).mean(), np.array(accT_mot_cls).mean()))

        if (epoch) % args.save_epoch == 0:
            states = {}
            record = {}
            states['cnn'] = cnn.state_dict()
            states['netE2'] = netE2.state_dict()
            states['netG_mot'] = netG_mot.state_dict()
            states['Dis_att'] = Dis_att.state_dict()
            states['Dis_mot'] = Dis_mot.state_dict()
            record['mot_record'] = mot_record
            record['label_record'] = label_record
            torch.save(states, os.path.join(args.log_dir, 'mot_gen_epoch' + str(epoch)))
            torch.save(record, os.path.join(args.log_dir, 'mot_record' + str(epoch)))
            print('params of epoch %d are saved' % (epoch))
            torch.cuda.empty_cache()


if __name__ == '__main__':
    args = mot_parse_args()
    global device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    run(device, args)

