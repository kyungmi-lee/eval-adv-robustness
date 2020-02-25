import os
import argparse
import copy

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import sys
sys.path.insert(0, './core')

# Model, data, adversarial utils
from model_utils import *
from data_loader import *
from count_flops import *
from adversarial_evaluation_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_on_trainset', default=False, action='store_true')
    parser.add_argument('--train_val_split', default=False, action='store_true')
    
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--model_type', default='wrn', type=str)
    parser.add_argument('--scale_factor', default=1, type=float)
    parser.add_argument('--spectral_norm', action='store_true', default=False)
    parser.add_argument('--no_batch_norm', default=False, action='store_true')
    
    parser.add_argument('--transfer_from', type=str)
    parser.add_argument('--transfer_model_type', default='wrn', type=str)
    parser.add_argument('--transfer_scale_factor', default=1, type=float)
    parser.add_argument('--transfer_spectral_norm', default=False, action='store_true')
    parser.add_argument('--transfer_no_batch_norm', default=False, action='store_true')
    
    parser.add_argument('--attack_type', default='pgd', type=str)
    parser.add_argument('--eps', default=8.0/255.0, type=float)
    parser.add_argument('--nb_iter', default=7, type=int)
    parser.add_argument('--eps_iter', default=2.0/255.0, type=float)
    parser.add_argument('--nb_random_start', default=1, type=int)
    
    parser.add_argument('--zero_compensation', default='none', type=str)
    parser.add_argument('--temperature', default=100., type=float)
    parser.add_argument('--targeted_to', default='random', type=str)
    parser.add_argument('--bpda_compensation', default=False, action='store_true')
    parser.add_argument('--relu_sub', default='softplus', type=str)
    parser.add_argument('--relu_sub_slope', default=2, type=float)
    parser.add_argument('--maxpool_sub_p', default=5, type=int)
    parser.add_argument('--second_order_init_method', default='rand', type=str)
    
    parser.add_argument('--count_flops', default=False, action='store_true')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    transform_test = transforms.Compose([transforms.ToTensor()])
    if args.eval_on_trainset:
         train_loader, val_loader = get_train_valid_loader('../data', args.batch_size, True, args.seed, 0.1 if \
                                                           args.train_val_split else 0, pin_memory=True)
           
         test_loader = train_loader
    else:
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=False, transform=transform_test),\
                                                  batch_size=args.batch_size, shuffle=False)
        
    model = get_models(args.model_type, args.load_model, args.scale_factor, device, dataset='cifar', \
                       spectral_norm=args.spectral_norm, batch_norm=not(args.no_batch_norm))
    
    if args.transfer_from is not None:
        prCyan("Black-box evaluation; source model: {}".format(args.transfer_from))
        transfer = get_models(args.transfer_model_type, args.transfer_from, args.transfer_scale_factor, device, dataset='cifar', \
                              spectral_norm=args.transfer_spectral_norm, batch_norm=not(args.transfer_no_batch_norm)) 
        adversary, iterative = get_adversary(transfer, args.attack_type, args.eps, args.nb_iter, args.eps_iter, args.nb_random_start)
        evaluate_all(model, test_loader, adversary, device, \
                     args.zero_compensation, args.temperature, args.targeted_to,\
                     args.bpda_compensation, args.relu_sub, args.relu_sub_slope, 2.0, args.maxpool_sub_p,\
                     iterative, args.second_order_init_method, \
                     args.transfer_model_type, args.transfer_scale_factor, transfer)
    else:
        adversary, iterative = get_adversary(model, args.attack_type, args.eps, args.nb_iter, args.eps_iter, args.nb_random_start)
        evaluate_all(model, test_loader, adversary, device, \
                     args.zero_compensation, args.temperature, args.targeted_to, \
                     args.bpda_compensation, args.relu_sub, args.relu_sub_slope, 2.0, args.maxpool_sub_p, \
                     iterative, args.second_order_init_method, \
                     args.model_type, args.scale_factor, None)
        
    if args.count_flops:
        model = model.cpu()
        print("Counting the model parameters and FLOPS for a single pass...")
        total_param = print_model_param_nums(model=model)
        total_flops = count_model_param_flops(model=model, input_res=32)
    
if __name__ == '__main__':
    main()
    
    
    