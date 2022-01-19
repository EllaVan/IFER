import torch
from torch import nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class BasicConv2d_Ins(nn.Module):
    '''
    BasicConv2d module with InstanceNorm
    '''
    def __init__(self, in_planes, out_planes, kernal_size, stride, padding):
        super(BasicConv2d_Ins, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernal_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.InstanceNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x


class block32_Ins(nn.Module):
    def __init__(self, scale=1.0):
        super(block32_Ins, self).__init__()

        self.scale = scale

        self.branch0 = nn.Sequential(BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0))

        self.branch1 = nn.Sequential(
        BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
        BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(48, 64, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class resblock(nn.Module):
    '''
    residual block
    '''
    def __init__(self, n_chan):
        super(resblock, self).__init__()
        self.infer = nn.Sequential(*[
            nn.Conv2d(n_chan, n_chan, 3, 1, 1),
            nn.ReLU()
        ])

    def forward(self, x_in):
        self.res_out = x_in + self.infer(x_in)
        return self.res_out


class decoder(nn.Module):
    def __init__(self, Nc=512, GRAY=False):
        super(decoder, self).__init__()

        self.us1 = nn.Sequential(*[
            nn.ConvTranspose2d(Nc, 2048, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(2048),
            nn.ReLU(True),
        ])
        self.us2 = nn.Sequential(*[
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.ReLU(True),
        ])
        self.us3 = nn.Sequential(*[
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
        ])
        self.us4 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        ])
        self.us5 = nn.Sequential(*[
            nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ])
        if GRAY:
            self.us6 = nn.Sequential(*[
                nn.ConvTranspose2d(64, 1, 3, 1, 1, bias=False),
                nn.Sigmoid()
            ])
        else:
            self.us6 = nn.Sequential(*[
                nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
                nn.Sigmoid()
            ])

    def _make_layer(self, block, num_blocks, n_chan):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(n_chan))
        return nn.Sequential(*layers)

    def forward(self, feat):
        self.emb_in = feat
        # bsx512x7x7 -> bsx2048x14x14
        self.us1_out = self.us1(self.emb_in)
        # bsx2048x14x14 -> bsx1024x28x28
        self.us2_out = self.us2(self.us1_out)
        # bsx1024x28x28 -> bsx512x56x56
        self.us3_out = self.us3(self.us2_out)
        # bsx512x56x56 -> bsx256x112x112
        self.us4_out = self.us4(self.us3_out)
        # bsx256x112x112 -> bsxout_chanx224x224
        self.us5_out = self.us5(self.us4_out)
        # bsx256x112x112 -> bsxout_chanx224x224
        self.img = self.us6(self.us5_out)

        return self.img