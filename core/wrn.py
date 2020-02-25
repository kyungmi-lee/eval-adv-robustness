# code from https://github.com/kuangliu/pytorch-cifar
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, './core')

from spectral_norm import SpectralNorm

__all__ = ['wrn']

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, spectral_norm=False, spectral_iter=1, res_spec=False,\
                 bpda_sub_on=False, bpda_relu_replace_fn=None):
        super(BasicBlock, self).__init__()
        if not spectral_norm:
            self.bn1 = nn.BatchNorm2d(in_planes)
            layer_init(self.bn1)
            self.relu1 = nn.ReLU(inplace=True)
        else:
            # https://discuss.pytorch.org/t/why-relu-inplace-true-does-not-give-error-in-official-resnet-py-but-it-gives-error-in-my-code/21004/6
            # relu inplace operation directly on conv(or spectral-normed conv) can be problematic as we don't have batchnorm
            # conv layer needs output to compute gradient; no inplace operation
            self.relu1 = nn.ReLU()
        if bpda_sub_on:
            self.relu1 = copy.deepcopy(bpda_relu_replace_fn)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        layer_init(self.conv1)
        if spectral_norm:
            self.conv1 = SpectralNorm(self.conv1, power_iterations=spectral_iter)
        
        if not spectral_norm:
            self.bn2 = nn.BatchNorm2d(out_planes)
            layer_init(self.bn2)
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.relu2 = nn.ReLU()
        if bpda_sub_on:
            self.relu2 = copy.deepcopy(bpda_relu_replace_fn)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        layer_init(self.conv2)
        if spectral_norm:
            self.conv2 = SpectralNorm(self.conv2, power_iterations=spectral_iter)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        if self.convShortcut:
            layer_init(self.convShortcut)
            if spectral_norm and res_spec:
                self.convShortcut = SpectralNorm(self.convShortcut, power_iterations=spectral_iter)
        self.spectral_norm = spectral_norm
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(x) if self.spectral_norm else  self.relu1(self.bn1(x))
        else:
            out = self.relu1(x) if self.spectral_norm else self.relu1(self.bn1(x))
        out =  self.relu2((self.conv1(out if self.equalInOut else x))) if self.spectral_norm else self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, \
                 spectral_norm=False, spectral_iter=1, res_spec=False,\
                 bpda_sub_on=False, bpda_relu_replace_fn=None):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, \
                                      spectral_norm, spectral_iter, res_spec,\
                                      bpda_sub_on, bpda_relu_replace_fn)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, \
                    spectral_norm, spectral_iter, res_spec, bpda_sub_on, bpda_relu_replace_fn):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, \
                                spectral_norm, spectral_iter, res_spec, bpda_sub_on, bpda_relu_replace_fn))
            #prev_layer = layers[-1].conv2
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

def layer_init(layer):
    if isinstance(layer, nn.Conv2d):
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.bias.data.zero_()

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, spectral_norm=False, spectral_iter=1, res_spec=False,\
                 bpda_sub_on=False, bpda_relu_replace_fn=None):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        layer_init(self.conv1)
        if spectral_norm:
            self.conv1 = SpectralNorm(self.conv1, power_iterations=spectral_iter)
        #if spectral_norm:
        #self.conv1 = SpectralNorm(self.conv1)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, \
                                   spectral_norm, spectral_iter, res_spec, bpda_sub_on, bpda_relu_replace_fn)
        # 2nd block
        #print(self.block1)
        #print(self.block1.layer[-1])
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, \
                                   spectral_norm, spectral_iter, res_spec, bpda_sub_on, bpda_relu_replace_fn)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, \
                                   spectral_norm, spectral_iter, res_spec, bpda_sub_on, bpda_relu_replace_fn)
        # global average pooling and classifier
        if not spectral_norm:
            self.bn1 = nn.BatchNorm2d(nChannels[3])
            layer_init(self.bn1)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.ReLU()
        if bpda_sub_on:
            self.relu = copy.deepcopy(bpda_relu_replace_fn)
        self.fc = nn.Linear(nChannels[3], num_classes)
        layer_init(self.fc)
        self.nChannels = nChannels[3]
        
        #self.spectral_norm = spectral_norm
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #print(m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        """

        self.spectral_norm = spectral_norm

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(out) if self.spectral_norm else self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model
