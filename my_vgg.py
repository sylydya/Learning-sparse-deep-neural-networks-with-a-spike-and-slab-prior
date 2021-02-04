import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from mask_module import Mask_Linear, Mask_Conv2d, Mask_BatchNorm2d, Mask_BatchNorm1d, Mask_ReLU, Mask_MaxPool2d, Mask_Sequential, Mask_Dropout

from collections import OrderedDict, namedtuple





class ConvBNReLU(nn.Module):

    def __init__(self, nInputPlane, nOutputPlane):
        super(ConvBNReLU, self).__init__()
        self.conv = Mask_Conv2d(nInputPlane, nOutputPlane, kernel_size=3, stride=1, padding=1)
        self.bn = Mask_BatchNorm2d(nOutputPlane, eps=1e-3)
        self.relu = Mask_ReLU(inplace=True)

        self._masks = OrderedDict()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def to(self, *args, **kwargs):
        super(vgg_cifar_sparse, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for name, para in self.named_masks():
            para.data = para.to(device)

    def update_mask(self, user_mask):
        for name, para in self.named_masks():
            para.data = user_mask[name]

    def get_mask(self):
        total_mask = {}
        for name, para in self.named_masks():
            total_mask[name] = para.clone()
        return total_mask

class vgg_cifar_sparse(nn.Module):
    def __init__(self):
        super(vgg_cifar_sparse, self).__init__()
        layers = []
        layers.append(ConvBNReLU(3,64))
        layers.append(Mask_Dropout(p = 0.3))
        layers.append(ConvBNReLU(64,64))
        layers.append(Mask_MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))

        layers.append(ConvBNReLU(64,128))
        layers.append(Mask_Dropout(p = 0.4))
        layers.append(ConvBNReLU(128,128))
        layers.append(Mask_MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))

        layers.append(ConvBNReLU(128, 256))
        layers.append(Mask_Dropout(p=0.4))
        layers.append(ConvBNReLU(256, 256))
        layers.append(Mask_Dropout(p=0.4))
        layers.append(ConvBNReLU(256, 256))
        layers.append(Mask_MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))

        layers.append(ConvBNReLU(256, 512))
        layers.append(Mask_Dropout(p=0.4))
        layers.append(ConvBNReLU(512, 512))
        layers.append(Mask_Dropout(p=0.4))
        layers.append(ConvBNReLU(512, 512))
        layers.append(Mask_MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))


        layers.append(ConvBNReLU(512, 512))
        layers.append(Mask_Dropout(p=0.4))
        layers.append(ConvBNReLU(512, 512))
        layers.append(Mask_Dropout(p=0.4))
        layers.append(ConvBNReLU(512, 512))
        layers.append(Mask_MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))


        self.conv_block = Mask_Sequential(*layers)

        fc_layers = []
        fc_layers.append(Mask_Dropout(p=0.4))
        fc_layers.append(Mask_Linear(512, 512))
        fc_layers.append(Mask_BatchNorm1d(512))
        fc_layers.append(Mask_ReLU(inplace=True))
        fc_layers.append(Mask_Dropout(p=0.5))
        fc_layers.append(Mask_Linear(512, 10))
        self.fc_block = Mask_Sequential(*fc_layers)

        self._masks = OrderedDict()

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def to(self, *args, **kwargs):
        super(vgg_cifar_sparse, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for name, para in self.named_masks():
            para.data = para.to(device)

    def update_mask(self, user_mask):
        for name, para in self.named_masks():
            para.data = user_mask[name]

    def get_mask(self):
        total_mask = {}
        for name, para in self.named_masks():
            total_mask[name] = para.clone()
        return total_mask






