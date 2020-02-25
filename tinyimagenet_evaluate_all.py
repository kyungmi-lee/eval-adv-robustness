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

# DataLoader
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, './core')

# Model, data, adversarial utils
from model_utils import *

from adversarial_evaluation_utils import *
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_on_trainset', default=False, action='store_true')
    
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--model_type', default='wrn', type=str, help='wrn, vgg')
    parser.add_argument('--scale_factor', default=1, type=int)
    parser.add_argument('--vgg_depth', default=11, type=int)
    parser.add_argument('--vgg_batch_norm', default=False, action='store_true')
  
    parser.add_argument('--transfer_from', type=str)
    parser.add_argument('--transfer_model_type', default='wrn', type=str, help='wrn, vgg')
    parser.add_argument('--transfer_scale_factor', default=1, type=int)
    parser.add_argument('--transfer_vgg_depth', default=11, type=int)
    parser.add_argument('--transfer_vgg_batch_norm', default=False, action='store_true')
    
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
    parser.add_argument('--maxpool_sub_p', default=5, type=int)
    parser.add_argument('--second_order_init_method', default='rand', type=str)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    data_dir = '../data/tiny-imagenet-200'
    transform_test = transforms.Compose([transforms.ToTensor()])
    if args.eval_on_trainset:
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_test)
        test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    else:
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
        
    model = get_models(args.model_type, args.load_model, args.scale_factor, device, dataset='tinyimagenet', \
                       batch_norm=args.vgg_batch_norm, tinyimagenet_vgg_depth=args.vgg_depth)
    
    if args.transfer_from is not None:
        transfer = get_models(args.transfer_model_type, args.transfer_from, args.transfer_scale_factor, device, \
                              dataset='tinyimagenet', batch_norm=args.transfer_vgg_batch_norm, \
                              tinyimagenet_vgg_depth=args.transfer_vgg_depth) 
        adversary, iterative = get_adversary(transfer, args.attack_type, args.eps, args.nb_iter, args.eps_iter, args.nb_random_start)
        evaluate_all(model, test_loader, adversary, device, \
                     args.zero_compensation, args.temperature, args.targeted_to,\
                     args.bpda_compensation, args.relu_sub, args.relu_sub_slope, 2.0, args.maxpool_sub_p,\
                     iterative, args.second_order_init_method, \
                     args.transfer_model_type, args.transfer_scale_factor, transfer, \
                     'tinyimagenet', args.transfer_vgg_depth, args.transfer.vgg_batch_norm)
    else:
        adversary, iterative = get_adversary(model, args.attack_type, args.eps, args.nb_iter, args.eps_iter, args.nb_random_start)
        evaluate_all(model, test_loader, adversary, device, \
                     args.zero_compensation, args.temperature, args.targeted_to, \
                     args.bpda_compensation, args.relu_sub, args.relu_sub_slope, 2.0, args.maxpool_sub_p, \
                     iterative, args.second_order_init_method, \
                     args.model_type, args.scale_factor, None, \
                     'tinyimagenet', args.vgg_depth, args.vgg_batch_norm)
        
        
if __name__ == '__main__':
    main()
    
    
    