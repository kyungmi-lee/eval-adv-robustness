import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_norm import SpectralNorm

# Add path to advertorch folder
import sys
sys.path.insert(0, './advertorch')
sys.path.insert(0, './core')

from advertorch.bpda import BPDAWrapper

from simple_models import *
from wrn import WideResNet

import copy

# For simple-cifar
class bpda_wrapped_simple(cifar_conv4fc2):
    def __init__(self, net, scale_factor=1., spectral_norm=False, spectral_iter=1, batch_norm=True, p=5, relu='softplus',\
                 relu_slope=2.0, softplus_threshold=2.0):
        super(bpda_wrapped_simple, self).__init__(scale_factor, spectral_norm, spectral_iter, batch_norm)
        # copy weights
        if spectral_norm == False:
            for m, m_source in zip(self.parameters(), net.parameters()):
                m.data = copy.deepcopy(m_source.data)
        else:
            for m, m_source in zip(self.modules(), net.modules()):
                if isinstance(m, SpectralNorm):
                    m.module.weight_bar.data = copy.deepcopy(m_source.module.weight_bar.data)
                    m.module.weight_u.data = copy.deepcopy(m_source.module.weight_u.data)
                    m.module.weight_v.data = copy.deepcopy(m_source.module.weight_v.data)
                    if hasattr(m.module, 'bias'):
                        m.module.bias.data = copy.deepcopy(m_source.module.bias.data)
                if isinstance(m, nn.Linear):
                    if hasattr(m, 'weight'):
                        m.weight.data = copy.deepcopy(m_source.weight.data)
                        if hasattr(m, 'bias'):
                            m.bias.data = copy.deepcopy(m_source.bias.data)
        
        if batch_norm:
            # copy the running mean and running var
            self.bn1.running_mean = net.bn1.running_mean
            self.bn1.running_var = net.bn1.running_var
            self.bn2.running_mean = net.bn2.running_mean
            self.bn2.running_var = net.bn2.running_var
            self.bn3.running_mean = net.bn3.running_mean
            self.bn3.running_var = net.bn3.running_var
            self.bn4.running_mean = net.bn4.running_mean
            self.bn4.running_var = net.bn4.running_var 
            
        # Define BPDA Wrappers for ReLU and MaxPool
        if relu == 'elu':
            self.relu = BPDAWrapper(nn.ReLU(), forwardsub=nn.ELU())
        elif relu == 'celu':
            self.relu = BPDAWrapper(nn.ReLU(), forwardsub=nn.CELU(relu_slope))
        elif relu == 'pass':
            self.relu = BPDAWrapper(nn.ReLU(), forwardsub=lambda x: x)
        elif relu == 'softplus':
            self.relu = BPDAWrapper(nn.ReLU(), forwardsub=nn.Softplus(beta=relu_slope, threshold=softplus_threshold))
        elif relu == 'relu':
            self.relu = nn.ReLU()

        if p == 0:
            self.maxpool = nn.MaxPool2d(2, 2)
        else:
            self.maxpool = BPDAWrapper(nn.MaxPool2d(2, 2), forwardsub=nn.LPPool2d(p, 2, 2))
        #self.maxpool = nn.MaxPool2d(2, 2)
        #if spectral_norm:
        #    print(self)
        
    def forward(self, x):
        x = self.relu((self.conv1(x))) if (self.spectral_norm or (not self.batch_norm)) else \
            self.relu(self.bn1(self.conv1(x)))
        x = self.relu((self.conv2(x))) if (self.spectral_norm or (not self.batch_norm)) else \
            self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = self.relu((self.conv3(x))) if (self.spectral_norm or (not self.batch_norm)) else \
            self.relu(self.bn3(self.conv3(x)))
        x = self.relu((self.conv4(x))) if (self.spectral_norm or (not self.batch_norm)) else \
            self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
# For WRN
class bpda_wrapped_wrn(WideResNet):
    def __init__(self, net, depth, num_classes, widen_factor=1, dropRate=0.0, spectral_norm=False, spectral_iter=1, res_spec=False,\
                 relu_replace='softplus', relu_replace_slope_param=2.0, relu_replace_threshold=2.0):
        # Define relu replace function
        if relu_replace == 'softplus':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.Softplus(beta=relu_replace_slope_param,\
                                                                            threshold=relu_replace_threshold))
            bpda_sub_on = True
        elif relu_replace == 'elu':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.ELU())
            bpda_sub_on = True
        elif relu_replace == 'celu':
            relu_replace_fn = BPDAWrapper(nn.ReLU(), forwardsub=nn.CELU(relu_replace_slope_param))
            bpda_sub_on = True
        #elif relu_replace == 'relu':
        #    relu_replace_fn = nn.ReLU(inplace=True)
        #    bpda_sub_on = True
        else:
            relu_replace_fn = nn.ReLU()
            bpda_sub_on = False
        
        super(bpda_wrapped_wrn, self).__init__(depth, num_classes, int(widen_factor), dropRate, \
                                               spectral_norm, spectral_iter, res_spec, \
                                               bpda_sub_on=bpda_sub_on, bpda_relu_replace_fn=relu_replace_fn)
        
        # Debug
        #for i, (m, m_source) in enumerate(zip(self.modules(), net.modules())):
        #    print(i, type(m), type(m_source))
        # copy weights
        if spectral_norm == False:
            for m, m_source in zip(self.parameters(), net.parameters()):
                m.data = copy.deepcopy(m_source.data)
        else:
            for m, m_source in zip(self.modules(), net.modules()):
                if isinstance(m, SpectralNorm):
                    m.module.weight_bar.data = copy.deepcopy(m_source.module.weight_bar.data)
                    m.module.weight_u.data = copy.deepcopy(m_source.module.weight_u.data)
                    m.module.weight_v.data = copy.deepcopy(m_source.module.weight_v.data)
                    if hasattr(m.module, 'bias'):
                        m.module.bias = copy.deepcopy(m_source.module.bias)
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    if hasattr(m, 'weight'):
                        m.weight.data = copy.deepcopy(m_source.weight.data)
                        if hasattr(m, 'bias'):
                            m.bias = copy.deepcopy(m_source.bias)
                            
        # For batch norm
        for m, m_source in zip(self.modules(), net.modules()):
            if isinstance(m, nn.BatchNorm2d):
        #        print(m)
        #        print(hasattr(m, 'running_mean'))
                m.running_mean = copy.deepcopy(m_source.running_mean)
                m.running_var = copy.deepcopy(m_source.running_var)
                
        #if spectral_norm:
        #    print(self)
