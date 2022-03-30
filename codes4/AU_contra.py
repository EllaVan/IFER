import argparse
import os
import time
import sys
import shutil

import pandas as pd
import numpy as np
import math

from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from materials.glove import GloVe

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


def AU_parse_args():
    parser = argparse.ArgumentParser()

    # model dataset
    parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu choice')
    parser.add_argument('--dataset', type=str, default='FERPLUS',
                        choices=['RAF', 'AffectNet', 'FERPLUS', 'path'], help='dataset')
    parser.add_argument('--log_dir', type=str, default="save/FERPLUS/2022-02-25/au_contra")
    parser.add_argument('--ex_au_path', default='materials/AF_ex_au.csv')
    parser.add_argument('--mot_epoch', type=int, default=120, help='number of epochs of mot_gen')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start')
    parser.add_argument('--start_file', type=str, default=None, help='net state file')

    parser.add_argument('--end_epoch', type=int, default=500, help='number of training epochs')
    parser.add_argument('--save_epoch', type=int, default=50, help='save frequency')


    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='200,300,400',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--glove_path', default='/media/database/data4/wf/work01/'
                                                'materials/glove.6B.300d.txt')
    parser.add_argument('--au_action_path', default='materials/AU_action.txt')
    parser.add_argument('--au_feat_path', default='materials/au_embedding_dict')


    # temperature
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    if opt.start_epoch == 0:
        if os.path.exists(opt.log_dir) is True:
            shutil.rmtree(opt.log_dir)
            os.makedirs(opt.log_dir)
        else:
            os.makedirs(opt.log_dir)
    opt.mot_path = 'save/{}/2022-02-25/decoder0226/mot_record{}'.format(opt.dataset, opt.mot_epoch)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    return opt


def get_au_embedding(glove, au_action_path):
    au_description = {}
    with open(au_action_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            terms = line.split(':')
            au_description[terms[0]] = terms[1]
    au_name = list(au_description)
    au_vectors = []
    au_embedding = {}
    for i in range(len(au_name)):
        au_vectors.append(glove[au_description[au_name[i]]])
    au_vectors = torch.stack(au_vectors)
    au_vectors = F.normalize(au_vectors)
    for i in range(len(au_name)):
        au_embedding[au_name[i]] = au_vectors[i]
    return au_embedding


def get_data(args):
    # motion features
    mot_gen_file = torch.load(args.mot_path)
    mot_feats = torch.cat(mot_gen_file['mot_record'], dim=0).cpu().detach()
    labels_tmp = torch.cat(mot_gen_file['label_record'], dim=0).cpu().detach()
    labels = labels_tmp.numpy().tolist()

    # AU initial semantic features
    if args.au_feat_path is None:
        glove = GloVe(args.glove_path)
        au_embedding = get_au_embedding(glove, args.au_action_path)
    else:
        au_embedding = torch.load(args.au_feat_path)

    au_vectors = []
    au_name = list(au_embedding)
    for i in range(len(au_embedding)):
        au_vectors.append(au_embedding[au_name[i]])
    au_tensors = torch.stack(au_vectors, dim=0)

    return mot_feats, labels, au_tensors


class feat_set(data.Dataset):
    def __init__(self, phase, feats, labels):
        self.phase = phase
        self.feats = feats
        self.labels = labels

    def __getitem__(self, idx):
        mot_feat = self.feats[idx]
        label = self.labels[idx]
        return mot_feat, label, idx

    def __len__(self):
        return self.feats.shape[0]


def get_ex_au(ex_au_path):
    data_ex_au = pd.read_csv(ex_au_path)
    exs = list(data_ex_au.iloc[:, 0])
    ex_au = np.array(data_ex_au.iloc[:, 1:])
    return exs, ex_au


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


class SupConLoss(nn.Module):
    def __init__(self, ex_au_path, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.ex_au_path = ex_au_path
        self.exs, self.ex_au = self.get_ex_au(self.ex_au_path)

    def get_ex_au(self, ex_au_path):
        data_ex_au = pd.read_csv(ex_au_path)
        exs = list(data_ex_au.iloc[:, 0])
        ex_au = np.array(data_ex_au.iloc[:, 1:])
        return exs, ex_au

    def forward(self, labels, mot_feats, au_feats_all):

        batch_size = mot_feats.shape[0]
        for sample_i in range(batch_size):
            label_ex = labels[sample_i]
            if label_ex != 0:
                au_feature = []
                au_weight = []
                mot_feature = mot_feats[sample_i]
                sample_ex_au = self.ex_au[label_ex]
                sample_au = []
                for au_i in range(len(sample_ex_au)):
                    if sample_ex_au[au_i] != 0:
                        au_feature.append(au_feats_all[au_i])
                        au_weight.append(sample_ex_au[au_i])
                        sample_au.append(au_i)

                contrast_count = len(au_weight)
                au_feats_pos = au_feats_all[sample_au]
                contrast_feature = torch.cat(torch.unbind(au_feats_pos, dim=0), dim=0)
                # contrast_feature = au_feats_pos

                loss = 0

                mot_dot_au_pos = []
                mot_dot_au_neg = 0
                for j in range(len(sample_ex_au)):
                    if sample_ex_au[j] != 0:
                        mot_dot_au_pos.append(torch.div(
                            torch.matmul(mot_feature, au_feats_all[j]),
                            self.temperature))
                    else:
                        mot_dot_au_neg += torch.exp(
                            torch.div(torch.matmul(mot_feature, au_feats_all[j]),
                                      self.temperature))
                denominator = torch.log(mot_dot_au_neg)
                logits = 0
                for k in range(len(mot_dot_au_pos)):
                    logits += mot_dot_au_pos[k] - denominator
                loss += logits / len(mot_dot_au_pos)

        loss /= labels.shape[0]
        return loss


class AUEqDimension(nn.Module):
    def __init__(self):
        super(AUEqDimension, self).__init__()
        self.fc1 = nn.Linear(300, 512)
        self.fc2 = nn.Linear(512, 512)
        self.apply(weights_init)

    def forward(self, x):
        self.h1 = self.fc1(x)
        self.h2 = self.fc2(self.h1)
        return self.h2


def set_model(args):
    criterion = SupConLoss(args.ex_au_path, temperature=args.temp)
    model = AUEqDimension()

    # enable synchronized Batch Normalization
    # if args.syncBN:
    #     model = apex.parallel.convert_syncbn_model(model)

    # parallel gpu training
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         model.encoder = torch.nn.DataParallel(model.encoder)
    #     model = model.cuda()
    #     criterion = criterion.cuda()
    #     cudnn.benchmark = True

    return model, criterion


def train(args, device, train_loader, au_tensors, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    au_tensors = Variable(au_tensors.to(device), requires_grad=True)
    exs, ex_au = get_ex_au(args.ex_au_path)

    for idx, (mot_feats, labels, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = labels.shape[0]
        mot_feats = mot_feats.to(device)
        labels = labels.to(device)
        au_tensors = au_tensors.to(device)
        model = model.to(device)

        # warm-up learning rate
        # warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        au_features = Variable(model(au_tensors), requires_grad=True)

        loss = criterion(labels, mot_feats, au_features)
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        print('Train: [{0}/{1}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, args.end_epoch, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        return losses.avg


def main():
    args = AU_parse_args()
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(args.log_dir)

    mot_feats, labels, au_embedding = get_data(args)
    phase = 'train'
    train_dataset = feat_set(phase, mot_feats, labels)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    model, criterion = set_model(args)
    optimizer = set_optimizer(args, model) # 对model的参数优化,但是现在还没有model

    # training routine
    for epoch in range(1, args.end_epoch + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        loss = train(args, device, train_loader, au_embedding, model, criterion, optimizer, epoch)

        # tensorboard logger
        writer.add_scalar("loss", loss, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.save_epoch == 0:
            save_file = os.path.join(
                args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

        # save the last model
    save_file = os.path.join(
        args.save_folder, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)


if __name__ == '__main__':
    # main()
    file = torch.load('/media/database/data4/wf/IFER/codes/codes4/save/RAF/2022-03-25/encoder/net_state/epoch200',
                      map_location='cpu')