from __future__ import absolute_import

'''Resnet for cifar dataset. 
Ported form 
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from mask_module import Mask_Linear, Mask_Conv2d, Mask_BatchNorm2d, Mask_ReLU, Mask_AvgPool2d, Mask_Sequential

from collections import OrderedDict, namedtuple


__all__ = ['resnet']

def mask_conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Mask_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = mask_conv3x3(inplanes, planes, stride)
        self.bn1 = Mask_BatchNorm2d(planes)
        self.relu = Mask_ReLU(inplace=True)
        self.conv2 = mask_conv3x3(planes, planes)
        self.bn2 = Mask_BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self._masks = OrderedDict()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Mask_Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Mask_BatchNorm2d(planes)
        self.conv2 = Mask_Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = Mask_BatchNorm2d(planes)
        self.conv3 = Mask_Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = Mask_BatchNorm2d(planes * 4)
        self.relu = Mask_ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self._masks = OrderedDict()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = Mask_Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = Mask_BatchNorm2d(16)
        self.relu = Mask_ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = Mask_AvgPool2d(8)
        self.fc = Mask_Linear(64 * block.expansion, num_classes)

        self._masks = OrderedDict()

        for m in self.modules():
            if isinstance(m, Mask_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Mask_BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Mask_Sequential(
                Mask_Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Mask_BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Mask_Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def to(self, *args, **kwargs):
        super(ResNet, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for name, para in self.named_masks():
            para.data = para.to(device)



class ResNet_sparse(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(ResNet_sparse, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = Mask_Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = Mask_BatchNorm2d(16)
        self.relu = Mask_ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = Mask_AvgPool2d(8)
        self.fc = Mask_Linear(64 * block.expansion, num_classes)

        self._masks = OrderedDict()

        for m in self.modules():
            if isinstance(m, Mask_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Mask_BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Mask_Sequential(
                Mask_Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Mask_BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Mask_Sequential(*layers)

    def forward(self, x):


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def to(self, *args, **kwargs):
        super(ResNet_sparse, self).to(*args, **kwargs)
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





class Lenet5_sparse(nn.Module):
    def __init__(self):
        super(Lenet5_sparse, self).__init__()
        self.conv1 = Mask_Conv2d(1, 20, 5, 1)
        self.conv2 = Mask_Conv2d(20, 50, 5, 1)
        self.fc1 = Mask_Linear(4*4*50, 500)
        self.fc2 = Mask_Linear(500, 10)

        self._masks = OrderedDict()

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def to(self, *args, **kwargs):
        super(Lenet5_sparse, self).to(*args, **kwargs)
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



class Lenet_300_100_sparse(nn.Module):
    def __init__(self):
        super(Lenet_300_100_sparse, self).__init__()
        self.fc1 = Mask_Linear(784, 300)
        self.fc2 = Mask_Linear(300, 100)
        self.fc3 = Mask_Linear(100, 10)

        self._masks = OrderedDict()

    def forward(self, x):
        x = x.view(x.size(0), 28*28)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def to(self, *args, **kwargs):
        super(Lenet_300_100_sparse, self).to(*args, **kwargs)
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




def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

