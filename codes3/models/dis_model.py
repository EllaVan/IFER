import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


#Encoder
class Encoder1(nn.Module):

    def __init__(self, args):
        super(Encoder1,self).__init__()
        layer_sizes = args.encoder_layer_sizes
        size_mot = args.size_mot
        size_att = args.size_att
        self.args = args
        if args.encoder_use_mot:
            layer_sizes[0] += size_mot
        self.fc1=nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3=nn.Linear(layer_sizes[-1], size_att*2)
        self.lrelu = nn.LeakyReLU(0.02, True)
        self.linear_means = nn.Linear(size_att*2, size_att)
        self.linear_log_var = nn.Linear(size_att*2, size_att)
        self.apply(weights_init)

    def forward(self, res, mot=None):
        x = res
        if self.args.encoder_use_mot:
            x = torch.cat((res, mot), dim=-1)
        x1 = self.lrelu(self.fc1(x))
        x2 = self.lrelu(self.fc3(x1))
        means = self.linear_means(x2)
        log_vars = self.linear_log_var(x2)
        return means, log_vars


class Encoder2(nn.Module):

    def __init__(self, args):
        super(Encoder2,self).__init__()
        layer_sizes = args.encoder_layer_sizes
        size_mot = args.size_mot
        size_att = args.size_att
        self.args = args
        if args.encoder_use_mot:
            layer_sizes[0] += size_mot
        self.fc1=nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3=nn.Linear(layer_sizes[-1], size_att)
        self.lrelu = nn.LeakyReLU(0.02, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, res, mot=None):
        x = res
        if self.args.encoder_use_mot:
            x = torch.cat((res, mot), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.sigmoid(self.fc3(x))
        eps = x
        return eps


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        layer_sizes = args.decoder_layer_sizes
        size_mot = args.size_mot
        size_att = args.size_att
        input_size = size_mot + size_att
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.02, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def _forward(self, att, mot=None):
        z = torch.cat((att, mot), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.fc3(x1)
        # x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x

    def forward(self, att, mot=None, a1=None, feedback_layers=None):
        if feedback_layers is None:
            return self._forward(att,mot)
        else:
            z = torch.cat((att, mot), dim=-1)
            x1 = self.lrelu(self.fc1(z))
            feedback_out = x1 + a1*feedback_layers
            # x = self.sigmoid(self.fc3(feedback_out))
            x = self.fc3(feedback_out)
            return x


class Generator_mot(nn.Module):
    def __init__(self, args):
        super(Generator_mot, self).__init__()
        layer_sizes = args.decoder_layer_sizes
        size_res = args.size_res
        size_att = args.size_att
        input_size = size_res+ size_att
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.02, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def _forward(self, res, att=None):
        z = torch.cat((res, att), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.fc3(x1)
        # x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x

    def forward(self, att, mot=None, a1=None, feedback_layers=None):
        if feedback_layers is None:
            return self._forward(att,mot)
        else:
            z = torch.cat((att, mot), dim=-1)
            x1 = self.lrelu(self.fc1(z))
            feedback_out = x1 + a1*feedback_layers
            x = self.sigmoid(self.fc3(feedback_out))
            return x


class Discriminator_D1(nn.Module):
    def __init__(self, args, cls_num):
        super(Discriminator_D1, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, cls_num)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x):
        self.hidden = self.lrelu(self.fc1(x))
        h = self.fc2(self.hidden)
        return h


def calc_gradient_penalty(netD, real_data, fake_data, input_att, opt, device):
    alpha = torch.rand(real_data.shape[0], 1).to(device)
    alpha = alpha.expand(real_data.size()).to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size()).to(device)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty