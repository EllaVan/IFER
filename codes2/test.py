import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import os.path as osp

from config import opt
from materials import image_feature, image_featureGZ
import util
from models import model
from models.resnet import make_resnet18_base, ResNet
from lib import WeightedL1, calc_gradient_penalty, loss_fn, mse_loss, contrastive_loss, get_p_loss
from lib import yz_disentangle_loss, zx_disentangle_loss, yx_disentangle_loss, zy_disentangle_loss, unconstrained_z_loss, get_p_loss
from tcvae.tcvae import anneal_kl
from survae.distributions.conditional.normal import ConditionalNormal


global device
# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.manualSeed)

# Dimension reduction on attributes,是否要对属性特征降维
if opt.pca_attribute != 0:
    opt.attSize = opt.pca_attribute

exp_name = util.get_exp_name(opt)
print("Test Experiment: %s" % exp_name)

# state = torch.load(osp.join('runs/'+exp_name, '149netState'))

# load data
if opt.dataset == 'RAF':
    train_loader, test_loader = image_feature.getRAFdata()
    semanticfile = 'materials/RAFsemantic'
elif opt.dataset == 'AffcetNet':
    train_loader, test_loader = image_feature.getAffectdata()
    semanticfile = 'materials/AFsemantic'
elif opt.dataset == 'FERPLUS':
    train_loader, test_loader = image_feature.getFERdata()
    semanticfile = 'materials/AFsemantic'

netS = None
log_to_file = True and (not opt.debug)

flag_not = 0
train_size = 0
num_classes = 17

classifier = nn.Linear(512, num_classes)
classifier.to(device)
classifier.eval()

classifier_y = nn.Linear(opt.attSize, num_classes)
classifier_y.to(device)
classifier_y.eval()

loss_eval = nn.CrossEntropyLoss()
loss_eval = loss_eval.to(device)
ave_loss = []
ave_acc = []
ave_acc_y = []
ave_loss_y = []
best_ave_acc = 0
best_ave_acc_y = 0
best_epoch_y = 0

# log_dir = "runs/%s" % (exp_name)
# writer = SummaryWriter(log_dir)

eval_nodes = train_loader.dataset.test_nodes
for epoch_i in range(1, 18):
    print(str(epoch_i*10))
    state = torch.load(osp.join('runs/' + exp_name, str(epoch_i*10)+'netState'))
    cnn = make_resnet18_base()
    sd = cnn.state_dict()
    sd.update(state['cnn'])
    cnn.load_state_dict(sd)
    cnn.to(device)

    netE = model.Encoder(opt)  # 编码器
    netE.load_state_dict(state['netE'])
    netE = netE.to(device)
    # netG = model.Generator(opt).to(device)    # 生成器
    # netG.load_state_dict(state['netG'])
    netD = model.Discriminator_D1(opt)  # 辨别器
    netD.load_state_dict(state['netD'])
    netD = netD.to(device)
    netF = model.Feedback(opt)
    netF.load_state_dict(state['netF'])
    netF = netF.to(device)
    netDec = model.AttDec(opt, opt.attSize)
    netDec.load_state_dict(state['netDec'])
    netDec = netDec.to(device)

    torch.cuda.empty_cache()
    # netG.eval()
    netDec.eval()
    netF.eval()
    netE.eval()
    netD.eval()
    cnn.eval()
    for label_i, label in enumerate(eval_nodes):
        torch.cuda.empty_cache()
        subloader = test_loader[label_i]
        acc = []
        acc_y = []
        # evaluate(device, subloader, eval_nodes[label_i])
        for i, (data, label, index) in enumerate(subloader, 1):
            if data is None:
                continue
            else:
                with torch.no_grad():
                    data = data.to(device)
                    label = (label-6).to(device)
                    feature = cnn(data)
                    logits = classifier(feature)
                    feature = feature.to(device)
                    feature_y = netDec(feature)
                    feature_y = feature_y.to(device)
                    logits_y = classifier_y(feature_y)
                    ave_loss.append(util.loss_to_float(loss_eval(logits, label)))
                    ave_loss_y.append(util.loss_to_float(loss_eval(logits_y, label)))
                    _, pred = torch.max(logits, dim=1)
                    _, pred_y = torch.max(logits_y, dim=1)
                    tmp = torch.eq(pred, label).type(torch.FloatTensor).mean().item()
                    acc.append(tmp)
                    ave_acc.append(tmp)
                    tmp_y = torch.eq(pred_y, label).type(torch.FloatTensor).mean().item()
                    acc_y.append(tmp_y)
                    ave_acc_y.append(tmp_y)
        print("Average acc of origin input feature of %s is %.4f, and the acc of disentangled feature is %.4f"
              % (eval_nodes[label_i], np.mean(acc), np.mean(acc_y)))
    # writer.add_scalar("ave_acc", np.mean(ave_acc), epoch_i*10)
    # writer.add_scalar("ave_acc_y", np.mean(ave_acc_y), epoch_i*10)
    # writer.add_scalar("ave_loss", np.mean(ave_loss), epoch_i*10)
    # writer.add_scalar("ave_loss_y", np.mean(ave_loss_y), epoch_i*10)
    print('Average acc of origin input feature of %s dataset is %.4f, and the acc of disentangled feature is %.4f'
          % (opt.dataset, np.mean(ave_acc), np.mean(ave_acc_y)))
    if np.mean(ave_acc) > best_ave_acc:
        best_ave_acc = np.mean(ave_acc)
    if np.mean(ave_acc_y) > best_ave_acc_y:
        best_ave_acc_y = np.mean(ave_acc_y)
        best_epoch_y = epoch_i*10
    print('\n')

print('Best acc of disentangled feature is %.4f, and is obtained at %s epoch'
      % (best_ave_acc_y, str(epoch_i*10)))

