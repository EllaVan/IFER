import argparse
import shutil
import os.path as osp
import os
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np

from models.resnet import ResNet
from materials import image_feature_nN


def set_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('using gpu {}'.format(gpu))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', default='CK+/nN')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--nepoch', default=40)
    parser.add_argument('--save_epoch', default=10)
    args = parser.parse_args()

    set_gpu(args.gpu)
    save_path = args.save_path
    if os.path.exists(save_path) is True:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    train_loader, test_loader = image_feature_nN.getCKdata()
    num_cls = len(train_loader.dataset.nodes)
    model = ResNet('resnet18', num_cls)
    pre_state_dict = torch.load('../materials/resnet18-base.pth')
    new_state_dict = {}
    for k, v in model.resnet_base.state_dict().items():
        if k in pre_state_dict.keys() and k != 'conv1.weight':
            new_state_dict[k] = pre_state_dict[k]  # 如果原模型的层也在新模型的层里面， 那新模型就加载原先训练好的权重
    model.resnet_base.load_state_dict(new_state_dict, False)

    model = model.cuda()
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss().cuda()

    keep_ratio = 0.9975
    trlog = {}
    trlog['loss'] = []
    trlog['acc'] = []
    sum = 0

    for epoch in range(1, args.nepoch+1):

        ave_loss = None
        ave_acc = []
        flag_not = 0

        for i, (data, label, index) in enumerate(train_loader, 1):
            if data is None:
                flag_not += 1
                continue
            else:
                data = data.cuda()
                label = label.cuda()
                sum += data.size(0)

                logits = model(data)
                loss = loss_fn(logits, label)

                _, pred = torch.max(logits, dim=1)
                ave_acc += torch.eq(pred, label).type(torch.FloatTensor)

                ave_loss = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print('epoch {}, loss={:.4f}, acc={:.4f}'
              .format(epoch, ave_loss, np.array(ave_acc).mean()))

        trlog['loss'].append(ave_loss)
        trlog['acc'].append(ave_acc)
        trlog['not in the train'] = flag_not
        torch.save(trlog, osp.join(save_path, 'trlog'))

        if epoch % args.save_epoch == 0:
            torch.save(model.resnet_base.state_dict(),
                       osp.join(save_path, 'epoch-{}res.pth'.format(epoch)))
            torch.save(model.fc.state_dict(),
                       osp.join(save_path, 'epoch-{}fc.pth'.format(epoch)))