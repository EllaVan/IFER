import torch
from resnet import *
file = torch.load('materials/resnet18-5c106cde.pth')
w = file['fc.weight'].data
b = file['fc.bias'].data
file.pop('fc.weight')
file.pop('fc.bias')
res18 = resnet18(pretrained=False, in_channels=3, fc_size=2048, out_dim=256)
sd = res18.state_dict()
sd_2 = {k: v for k, v in sd.items() if 'num_batches_tracked' not in k}
sd_2.update(file)
res18.load_state_dict(sd_2)