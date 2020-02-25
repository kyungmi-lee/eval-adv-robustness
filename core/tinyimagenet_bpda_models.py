import torch
import torch.nn as nn
import torch.nn.functional as F

# Add path to advertorch folder
import sys
sys.path.insert(0, './advertorch')
sys.path.insert(0, './core')

from advertorch.bpda import BPDAWrapper

# Models
from tinyimagenet_wrn import *
from tinyimagenet_vgg import *

import copy

class bpda_wrapped_tinyimagenet_wrn(ResNet):
    def __init__(self, net, width, relu_replace='softplus', relu_replace_slope_param=2.0, relu_replace_threshold=2.0):
        super(bpda_wrapped_tinyimagenet_wrn, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=200, width_per_group=width*64)
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
            if isinstance(m, Bottleneck):
                m.relu = copy.deepcopy(relu_replace_fn)
        
        # Copy BN statistics
        for m, m_source in zip(self.modules(), net.modules()):
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean = copy.deepcopy(m_source.running_mean)
                m.running_var = copy.deepcopy(m_source.running_var)
                
        
class bpda_wrapped_vgg(VGG):
    def __init__(self, net, width, depth, batch_norm,\
                 relu_replace='softplus', relu_replace_slope_param=2.0, relu_replace_threshold=2.0,\
                 maxpool_sub_p=0):
        if relu_replace == 'softplus':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.Softplus(beta=relu_replace_slope_param,\
                                                                            threshold=relu_replace_threshold))
        elif relu_replace == 'elu':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.ELU())
        elif relu_replace == 'celu':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.CELU(relu_replace_slope_param))
        else:
            relu_replace_fn = nn.ReLU()
            
        if maxpool_sub_p == 0:
            maxpool = nn.MaxPool2d(2, 2)
        else:
            maxpool = BPDAWrapper(nn.MaxPool2d(2, 2), forwardsub=nn.LPPool2d(maxpool_sub_p, 2, 2))
            
        features = make_layers(get_config(depth, width), batch_norm=batch_norm, bpda_sub=True,\
                               maxpool_sub=maxpool, relu_sub=relu_replace_fn)
        
        super(bpda_wrapped_vgg, self).__init__(features, init_weights=False)
        # Replace relu in classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            copy.deepcopy(relu_replace_fn),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            copy.deepcopy(relu_replace_fn),
            nn.Dropout(),
            nn.Linear(1024, 200)
        )
        
        # Copy weights
        for m, m_source in zip(self.parameters(), net.parameters()):
            m.data = copy.deepcopy(m_source.data)
            
        # Copy BN statistics
        for m, m_source in zip(self.modules(), net.modules()):
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean = copy.deepcopy(m_source.running_mean)
                m.running_var = copy.deepcopy(m_source.running_var)