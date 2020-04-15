import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import AlexNet, ResNet
from torchvision.models.resnet import Bottleneck

# Add path to advertorch folder
import sys
sys.path.insert(0, './advertorch')
sys.path.insert(0, './places365_pretrained')

from advertorch.bpda import BPDAWrapper

# Models
from wideresnet import ResNet as WideResNet
from wideresnet import BasicBlock as WideBasicBlock

import copy

class bpda_wrapped_places365_resnet18(WideResNet):
    def __init__(self, net, relu_replace='softplus', relu_replace_slope_param=2.0, relu_replace_threshold=2.0):
        super(bpda_wrapped_places365_resnet18, self).__init__(WideBasicBlock, [2, 2, 2, 2], num_classes=365)
        if relu_replace == 'softplus':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.Softplus(beta=relu_replace_slope_param,\
                                                                            threshold=relu_replace_threshold))
        elif relu_replace == 'elu':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.ELU())
        elif relu_replace == 'celu':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.CELU(relu_replace_slope_param))
        else:
            relu_replace_fn = nn.ReLU()
            
        # Copy weights
        for m, m_source in zip(self.parameters(), net.parameters()):
            m.data = copy.deepcopy(m_source.data)
            
        # Replace relu with relu_replace_fn
        self.relu = copy.deepcopy(relu_replace_fn)
        for m in self.modules():
            if isinstance(m, WideBasicBlock):
                m.relu = copy.deepcopy(relu_replace_fn)
        
        # Copy BN statistics
        for m, m_source in zip(self.modules(), net.modules()):
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean = copy.deepcopy(m_source.running_mean)
                m.running_var = copy.deepcopy(m_source.running_var)
                
                
class bpda_wrapped_places365_resnet50(ResNet):
    def __init__(self, net, relu_replace='softplus', relu_replace_slope_param=2.0, relu_replace_threshold=2.0, maxpool_sub_p=5):
        super(bpda_wrapped_places365_resnet50, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=365)
        if relu_replace == 'softplus':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.Softplus(beta=relu_replace_slope_param,\
                                                                            threshold=relu_replace_threshold))
        elif relu_replace == 'elu':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.ELU())
        elif relu_replace == 'celu':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.CELU(relu_replace_slope_param))
        else:
            relu_replace_fn = nn.ReLU()
            
        if maxpool_sub_p > 0:
            maxpool_replace_fn = BPDAWrapper(nn.MaxPool2d(kernel_size=3, stride=1), forwardsub=nn.LPPool2d(maxpool_sub_p, 3, 1))
        else:
            maxpool_replace_fn = nn.MaxPool2d(kernel_size=3, stride=1)
            
        # Copy weights
        for m, m_source in zip(self.parameters(), net.parameters()):
            m.data = copy.deepcopy(m_source.data)
            # print(m.shape, m_source.shape)
            
        # Replace relu with relu_replace_fn
        self.relu = copy.deepcopy(relu_replace_fn)
        self.maxpool = copy.deepcopy(maxpool_replace_fn)
        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.relu = copy.deepcopy(relu_replace_fn)
        
        # Copy BN statistics
        for m, m_source in zip(self.modules(), net.modules()):
            # print(type(m), type(m_source))
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean = copy.deepcopy(m_source.running_mean)
                m.running_var = copy.deepcopy(m_source.running_var)
                
class bpda_wrapped_places365_alexnet(AlexNet):
    def __init__(self, net, relu_replace='softplus', relu_replace_slope_param=2.0, relu_replace_threshold=2.0, \
                 maxpool_sub_p=5):
        super(bpda_wrapped_places365_alexnet, self).__init__(num_classes=365)
        if relu_replace == 'softplus':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.Softplus(beta=relu_replace_slope_param,\
                                                                            threshold=relu_replace_threshold))
        elif relu_replace == 'elu':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.ELU())
        elif relu_replace == 'celu':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.CELU(relu_replace_slope_param))
        else:
            relu_replace_fn = nn.ReLU()
            
        if maxpool_sub_p > 0:
            maxpool_replace_fn = BPDAWrapper(nn.MaxPool2d(kernel_size=3, stride=2), forwardsub=nn.LPPool2d(maxpool_sub_p, 3, 2))
        else:
            maxpool_replace_fn = nn.MaxPool2d(kernel_size=3, stride=2)
            
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            copy.deepcopy(relu_replace_fn),
            copy.deepcopy(maxpool_replace_fn),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            copy.deepcopy(relu_replace_fn),
            copy.deepcopy(maxpool_replace_fn),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            copy.deepcopy(relu_replace_fn),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            copy.deepcopy(relu_replace_fn),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            copy.deepcopy(relu_replace_fn),
            copy.deepcopy(maxpool_replace_fn),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            copy.deepcopy(relu_replace_fn),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            copy.deepcopy(relu_replace_fn),
            nn.Linear(4096, 365),
        )
        
        # Copy weights
        for m, m_source in zip(self.parameters(), net.parameters()):
            m.data = copy.deepcopy(m_source.data)
