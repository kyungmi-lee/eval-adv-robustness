import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, './core')

from spectral_norm import SpectralNorm

class mnist_mlp(nn.Module):
    def __init__(self, scale_factor = 1., spectral_norm=False):
        super(mnist_mlp, self).__init__()
        self.fc1 = nn.Linear(784, int(300*scale_factor))
        self.fc2 = nn.Linear(int(300*scale_factor), int(100*scale_factor))
        self.fc3 = nn.Linear(int(100*scale_factor), 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        
class mnist_conv2fc2(nn.Module):
    def __init__(self, scale_factor = 1., spectral_norm=False):
        super(mnist_conv2fc2, self).__init__()
        self.conv1 = nn.Conv2d(1, int(scale_factor*2), 5, 1, 2)
        if spectral_norm:
            self.conv1 = SpectralNorm(self.conv1)
        self.conv2 = nn.Conv2d(int(scale_factor*2), int(scale_factor*4), 5, 1, 2)
        if spectral_norm:
            self.conv2 = SpectralNorm(self.conv2)
        self.fc1 = nn.Linear(7*7*int(scale_factor*4), 1024)
        self.fc2 = nn.Linear(1024, 10)

        #print(self)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class cifar_conv4fc2(nn.Module):
    def __init__(self, scale_factor = 1., spectral_norm=False, spectral_iter=1, batch_norm=True,\
                 replace_relu_with_softplus=False, replace_maxpool_with_avgpool=False,\
                 replace_maxpool_with_stride=False):
        super(cifar_conv4fc2, self).__init__()
        self.spectral_norm = spectral_norm
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(3, int(scale_factor*8), 3, 1, 1)
        if not self.spectral_norm and self.batch_norm:
            self.bn1 = nn.BatchNorm2d(int(scale_factor*8))
        elif self.spectral_norm:
            self.conv1 = SpectralNorm(self.conv1, power_iterations=spectral_iter)
        
        stride = 2 if replace_maxpool_with_stride else 1
        self.conv2 = nn.Conv2d(int(scale_factor*8), int(scale_factor*8), 3, stride, 1)
        if not self.spectral_norm and self.batch_norm:
            self.bn2 = nn.BatchNorm2d(int(scale_factor*8))
        elif self.spectral_norm:
            self.conv2 = SpectralNorm(self.conv2, power_iterations=spectral_iter)

        self.conv3 = nn.Conv2d(int(scale_factor*8), int(scale_factor*16), 3, 1, 1)
        if not self.spectral_norm and self.batch_norm:
            self.bn3 = nn.BatchNorm2d(int(scale_factor*16))
        elif self.spectral_norm:
            self.conv3 = SpectralNorm(self.conv3, power_iterations=spectral_iter)
        
        stride = 2 if replace_maxpool_with_stride else 1
        self.conv4 = nn.Conv2d(int(scale_factor*16), int(scale_factor*16), 3, stride, 1)
        if not self.spectral_norm and self.batch_norm:
            self.bn4 = nn.BatchNorm2d(int(scale_factor*16))
        elif self.spectral_norm:
            self.conv4 = SpectralNorm(self.conv4, power_iterations=spectral_iter)

        self.fc1 = nn.Linear(int(scale_factor*16)*8*8, int(scale_factor*128))
        if self.spectral_norm:
            self.fc1 = SpectralNorm(self.fc1, power_iterations=spectral_iter)
        self.fc2 = nn.Linear(int(scale_factor*128), 10)
        
        self.softplus = replace_relu_with_softplus
        self.avgpool = replace_maxpool_with_avgpool
        self.stride = replace_maxpool_with_stride

    def forward(self, x):
        if self.softplus == False and (self.avgpool == False and self.stride == False):
            x = F.relu((self.conv1(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn1(self.conv1(x)))
            x = F.relu((self.conv2(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2, 2)

            x = F.relu((self.conv3(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn3(self.conv3(x)))
            x = F.relu((self.conv4(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn4(self.conv4(x)))
            x = F.max_pool2d(x, 2, 2)

            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)
        elif self.softplus == True and (self.avgpool == False and self.stride == False):
            x = F.softplus((self.conv1(x)), threshold=2) if (self.spectral_norm or (not self.batch_norm)) \
                else F.softplus(self.bn1(self.conv1(x)), threshold=2)
            x = F.softplus((self.conv2(x)), threshold=2) if (self.spectral_norm or (not self.batch_norm)) \
                else F.softplus(self.bn2(self.conv2(x)), threshold=2)
            x = F.max_pool2d(x, 2, 2)

            x = F.softplus((self.conv3(x)), threshold=2) if (self.spectral_norm or (not self.batch_norm)) \
                else F.softplus(self.bn3(self.conv3(x)), threshold=2)
            x = F.softplus((self.conv4(x)), threshold=2) if (self.spectral_norm or (not self.batch_norm)) \
                else F.softplus(self.bn4(self.conv4(x)), threshold=2)
            x = F.max_pool2d(x, 2, 2)

            x = x.view(x.size(0), -1)
            x = F.softplus(self.fc1(x), threshold=2)
            return self.fc2(x)
        elif self.softplus == False and (self.avgpool == True and self.stride == False):
            x = F.relu((self.conv1(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn1(self.conv1(x)))
            x = F.relu((self.conv2(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn2(self.conv2(x)))
            x = F.avg_pool2d(x, 2, 2)

            x = F.relu((self.conv3(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn3(self.conv3(x)))
            x = F.relu((self.conv4(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn4(self.conv4(x)))
            x = F.avg_pool2d(x, 2, 2)

            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)
        elif self.softplus == False and (self.avgpool == False and self.stride == True):
            x = F.relu((self.conv1(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn1(self.conv1(x)))
            x = F.relu((self.conv2(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn2(self.conv2(x)))

            x = F.relu((self.conv3(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn3(self.conv3(x)))
            x = F.relu((self.conv4(x))) if (self.spectral_norm or (not self.batch_norm)) else F.relu(self.bn4(self.conv4(x)))

            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)
#         else:
#             x = F.softplus((self.conv1(x)), threshold=2) if (self.spectral_norm or (not self.batch_norm)) \
#                 else F.softplus(self.bn1(self.conv1(x)), threshold=2)
#             x = F.softplus((self.conv2(x)), threshold=2) if (self.spectral_norm or (not self.batch_norm)) \
#                 else F.softplus(self.bn2(self.conv2(x)), threshold=2)
#             x = F.avg_pool2d(x, 2, 2)

#             x = F.softplus((self.conv3(x)), threshold=2) if (self.spectral_norm or (not self.batch_norm)) \
#                 else F.softplus(self.bn3(self.conv3(x)), threshold=2)
#             x = F.softplus((self.conv4(x)), threshold=2) if (self.spectral_norm or (not self.batch_norm)) \
#                 else F.softplus(self.bn4(self.conv4(x)), threshold=2)
#             x = F.avg_pool2d(x, 2, 2)

#             x = x.view(x.size(0), -1)
#             x = F.softplus(self.fc1(x), threshold=5)
#             return self.fc2(x)


