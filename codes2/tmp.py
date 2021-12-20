import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import os.path as osp
import time

from config import opt
from materials import neutral_image
import util
from models import model
from models.resnetDeconv import make_resnet18_deconv_base

from PIL import Image
from torchvision.transforms import ToPILImage

global device
# device = torch.device('cpu')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

f = torch.load('/media/database/data4/wf/IFER/codes2/mot_gaus/50deconv_states')
means = f['means']
cnn_deconv = make_resnet18_deconv_base()
sd = cnn_deconv.state_dict()
sd.update(f['cnn_deconv'])
cnn_deconv.load_state_dict(sd)
cnn_deconv.to(device)

tmp1 = means.t()
tmp2 = tmp1.expand(512, 49)
tmp3 = tmp2.unsqueeze(0)
tmp4 = tmp3.reshape(1, 512, 7,7)

# a = means.repeat(1, 49).reshape(means.size(0), means.size(1), 7, 7)
a = cnn_deconv(tmp4)
img = ToPILImage()(a[0])
img.save('/media/database/data4/wf/IFER/codes2/mot_gaus/img.png')