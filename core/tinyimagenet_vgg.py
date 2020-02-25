# https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg11
import torch
import torch.nn as nn
import copy

class VGG(nn.Module):

    def __init__(self, features, num_classes=200, width=1, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512 * width * 2 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024 , 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, bpda_sub=False, maxpool_sub=None, relu_sub=None):
    layers = []
    in_channels = 3
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2) if not bpda_sub else maxpool_sub
    act = nn.ReLU() if not bpda_sub else relu_sub
    for v in cfg:
        if v == 'M':
            layers += [copy.deepcopy(maxpool)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), copy.deepcopy(act)]
            else:
                layers += [conv2d, copy.deepcopy(act)]
            in_channels = v
    return nn.Sequential(*layers)

"""
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
"""

def get_config(depth, width):
    # 'A': 11, 'B': 13, 'D': 16, 'E': 19
    if depth == 11:
        return [64*width, 'M', 128*width, 'M', 256*width, 256*width, 'M', 512*width, 512*width, 'M', 512*width, 512*width, 'M']
    elif depth == 13:
        return [64*width, 64*width, 'M', 128*width, 128*width, 'M', 256*width, 256*width, \
                'M', 512*width, 512*width, 'M', 512*width, 512*width, 'M']
    elif depth == 16:
        return [64*width, 64*width, 'M', 128*width, 128*width, 'M', 256*width, 256*width, 256*width, 'M', \
                512*width, 512*width, 512*width, 'M', 512*width, 512*width, 512*width, 'M']
    elif depth == 19:
        return [64*width, 64*width, 'M', 128*width, 128*width, 'M', 256*width, 256*width, 256*width, 256*width, \
                'M', 512*width, 512*width, 512*width, 512*width, 'M', 512*width, 512*width, 512*width, 512*width, 'M']

def vgg(depth, width, batch_norm):
    model = VGG(make_layers(get_config(depth, width), batch_norm=batch_norm), width=width)
    return model