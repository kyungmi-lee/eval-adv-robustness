import os
import argparse
import copy

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# TorchVision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

# DataLoader
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, './core')
sys.path.insert(0, './places365_pretrained')

# Model, data, adversarial utils
from adversarial_evaluation_utils import *
import wideresnet
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    
    parser.add_argument('--model_type', type=str, help='resnet18, resnet50, alexnet')
    
    parser.add_argument('--attack_type', default='pgd', type=str)
    parser.add_argument('--eps', default=8.0/255.0, type=float)
    parser.add_argument('--nb_iter', default=7, type=int)
    parser.add_argument('--eps_iter', default=2.0/255.0, type=float)
    parser.add_argument('--nb_random_start', default=1, type=int)
    
    parser.add_argument('--zero_compensation', default='none', type=str)
    parser.add_argument('--temperature', default=100., type=float)
    parser.add_argument('--targeted_to', default='random', type=str)
    parser.add_argument('--bpda_compensation', default=False, action='store_true')
    parser.add_argument('--relu_sub', default='celu', type=str)
    parser.add_argument('--relu_sub_slope', default=2, type=float)
    parser.add_argument('--maxpool_sub_p', default=5, type=float)
    parser.add_argument('--second_order_init_method', default='rand', type=str)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    data_dir = '../data/Places365/val_256'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Scale(256), \
                                    transforms.CenterCrop(224), \
                                    transforms.ToTensor(),\
                                    normalize,\
                                   ])

    test_dataset = datasets.ImageFolder(data_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    if args.model_type == 'resnet18':
        model_file = './places365_pretrained/wideresnet18_places365.pth.tar'
        model = wideresnet.resnet18(num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.to(device)
    elif args.model_type == 'resnet50':
        arch = 'resnet50'
        model_file = './places365_pretrained/%s_places365.pth.tar' % arch
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.to(device)
    elif args.model_type == 'alexnet':
        arch = 'alexnet'
        model_file = './places365_pretrained/%s_places365.pth.tar' % arch
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.to(device)
    
    eps = [args.eps/0.229, args.eps/0.224, args.eps/0.225]
    eps_iter = [args.eps_iter/0.229, args.eps_iter/0.224, args.eps_iter/0.225]
    clip_min = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    clip_max = [(1.0-0.485)/0.229, (1.0-0.456)/0.224, (1.0-0.406)/0.225]
    
    adversary, iterative = get_adversary(model, args.attack_type, eps, args.nb_iter, eps_iter, args.nb_random_start, \
                                         clip_min, clip_max)
    evaluate_all(model, test_loader, adversary, device, \
                 args.zero_compensation, args.temperature, args.targeted_to, \
                 args.bpda_compensation, args.relu_sub, args.relu_sub_slope, 2.0, args.maxpool_sub_p, \
                 iterative, args.second_order_init_method, \
                 args.model_type, None, None, 'places365')
        
        
if __name__ == '__main__':
    main()
    
    
    