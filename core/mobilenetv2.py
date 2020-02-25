# code from https://github.com/kuangliu/pytorch-cifar
'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, './core')

from spectral_norm import SpectralNorm

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, spectral_norm):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        if spectral_norm:
            self.conv1 = SpectralNorm(self.conv1)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        if spectral_norm:
            self.conv2 = SpectralNorm(self.conv2)
        else:
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        if spectral_norm:
            self.conv3 = SpectralNorm(self.conv3)
        else:
            self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        self.spectral_norm = spectral_norm

    def forward(self, x):
        out = F.relu(self.conv1(x)) if self.spectral_norm else F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv2(out)) if self.spectral_norm else  F.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out) if self.spectral_norm else self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, width_mult=1., spectral_norm=False):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, int(32*width_mult), kernel_size=3, stride=1, padding=1, bias=False)
        if spectral_norm:
            self.conv1 = SpectralNorm(self.conv1)
        else:
            self.bn1 = nn.BatchNorm2d(int(32*width_mult))
        self.layers = self._make_layers(in_planes=int(32*width_mult), width_mult=width_mult, spectral_norm=spectral_norm)
        self.conv2 = nn.Conv2d(int(320*width_mult), int(1280*width_mult), kernel_size=1, stride=1, padding=0, bias=False)
        if spectral_norm:
            self.conv2 = SpectralNorm(self.conv2)
        else:
            self.bn2 = nn.BatchNorm2d(int(1280*width_mult))
        self.linear = nn.Linear(int(1280*width_mult), num_classes)
        self.spectral_norm = spectral_norm

    def _make_layers(self, in_planes, width_mult, spectral_norm):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, int(out_planes*width_mult), expansion, stride, spectral_norm))
                in_planes = int(out_planes*width_mult)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x)) if self.spectral_norm else F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.conv2(out)) if self.spectral_norm else F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
