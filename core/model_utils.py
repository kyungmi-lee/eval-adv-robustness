import torch

import sys
sys.path.insert(0, './core')

# Import models for CIFAR & SVHN
from wrn import WideResNet
from mobilenetv2 import MobileNetV2
from mobilenetv1 import MobileNetV1

# Import simple models for CIFAR & MNIST
from simple_models import *

# Import models for TinyImageNet
from tinyimagenet_wrn import TinyImageNet_WRN50
from tinyimagenet_vgg import vgg

def get_models(model_type, load_dir, scale_factor, device, dataset='cifar', \
               spectral_norm=False, spectral_iter=1, res_spec=False, batch_norm=True, \
               replace_relu_with_softplus=False, replace_maxpool_with_avgpool=False, \
               replace_maxpool_with_stride=False, tinyimagenet_vgg_depth=11):
    if dataset == 'cifar' or dataset == 'svhn':
        if model_type == 'wrn':
            model = WideResNet(depth=28, num_classes=10, widen_factor=int(scale_factor), spectral_norm=spectral_norm, spectral_iter=spectral_iter, res_spec=res_spec)
        elif model_type == 'mobile2':
            model = MobileNetV2(num_classes=10, width_mult=scale_factor, spectral_norm=spectral_norm)
        elif model_type == 'mobile1':
            model = MobileNetV1(num_classes=10, widen_factor=scale_factor)
        elif model_type == 'simple':
            model = cifar_conv4fc2(scale_factor=scale_factor, spectral_norm=spectral_norm, spectral_iter=spectral_iter, \
                                   batch_norm=batch_norm, replace_relu_with_softplus=replace_relu_with_softplus, \
                                   replace_maxpool_with_avgpool=replace_maxpool_with_avgpool, \
                                   replace_maxpool_with_stride=replace_maxpool_with_stride)
        else:
            print("Error: Unknown model type")
            model = None
    elif dataset == 'mnist':
        if model_type == 'simple':
            model = mnist_conv2fc2(scale_factor=scale_factor, spectral_norm=spectral_norm)
        elif model_type == 'mlp':
            model = mnist_mlp(scale_factor=scale_factor)
    elif dataset == 'tinyimagenet':
        if model_type == 'vgg':
            model = vgg(tinyimagenet_vgg_depth, int(scale_factor), batch_norm)
        elif model_type == 'wrn':
            model = TinyImageNet_WRN50(int(scale_factor))
    else:
        print("Error: Unknown dataset")
        model = None

    if load_dir is not None:
        model.load_state_dict(torch.load(load_dir))
    model = model.to(device)
    return model
