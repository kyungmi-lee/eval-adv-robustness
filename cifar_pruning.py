import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import sys
sys.path.insert(0, './core')

from adversarial_training_utils import *
from pruning_utils import *
from model_utils import *
from data_loader import *

def main():
    parser = argparse.ArgumentParser()
   
    # Basic Training Setup
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--log_interval', default=200, type=int)
    parser.add_argument('--train_val_split', default=False, action='store_true')

    # Adversarial Attack
    parser.add_argument('--disable_adv', action='store_true', default=False)
    parser.add_argument('--no_adv_eval', action='store_true', default=False)
    parser.add_argument('--attack_type', default='pgd', type=str)
    parser.add_argument('--eps', default=8.0/255.0, type=float)
    parser.add_argument('--nb_iter', default=7, type=int)
    parser.add_argument('--eps_iter', default=2.0/255.0, type=float)
    parser.add_argument('--nb_random_start', default=1, type=int)

    # Model
    parser.add_argument('--model_type', default='wrn', type=str, help='wrn | mobile2 | mobile1 | simple')
    parser.add_argument('--scale_factor', default=1, type=int)
    parser.add_argument('--no_batch_norm', default=False, action='store_true')

    # Training Hyperparameters
    parser.add_argument('--opt', default='sgd', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    
    # Pruning Flag
    parser.add_argument('--prune', action='store_true', default=False)
    parser.add_argument('--prune_method', default='global', type=str)
    parser.add_argument('--prune_all', action='store_true', default=False)
    parser.add_argument('--epochs', default=9, type=int, help='total number of pruning iterations')
    # When using --prune_ratio, each pruning iteration will remove args.prune_ratio / args.epochs (%) of weights each iteration
    parser.add_argument('--prune_ratio', default=0.9, type=float, help='final desired sparsity')
    parser.add_argument('--retrain_epochs', default=10, type=int)
    # When using --prune_fixed_percent, 
    # each pruning iteration will only keep args.pruned_ratio_per_step of **remaning** weights each iteration
    parser.add_argument('--prune_fixed_percent', default=False, action='store_true')
    parser.add_argument('--prune_ratio_per_step', default=0.75, type=float)

    # Save Model
    parser.add_argument('--load_model', default='mnist_adv_train.pt', type=str)
    parser.add_argument('--save_model', default='mnist_adv_train.pt', type=str)
    parser.add_argument('--prune_txt', default='prune.txt', type=str)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # For debugging at local desktop without GPU
    # device = torch.device("cpu")
    # If GPU is available:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), \
            transforms.RandomHorizontalFlip(),\
            transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    if args.train_val_split:
        train_loader, val_loader = get_train_valid_loader('../data', args.train_batch_size, True, args.seed, 0.1, pin_memory=True)
        test_loader = val_loader
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
            batch_size=args.train_batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transform_test),
            batch_size=args.test_batch_size, shuffle=False)
    
    model = get_models(args.model_type, args.load_model, args.scale_factor, device, dataset='cifar', batch_norm=not(args.no_batch_norm))
    adversary = get_adversary(args.attack_type, model, args)

    f = open(args.prune_txt, 'w')
    clnloss, clncorr, advloss, advcorr = test(model, test_loader, adversary, device, args)
    f.write('Before pruning: Cln loss: %.8f, Cln Acc: %.3f\n' % (clnloss, 1. * clncorr/len(test_loader.dataset)))
    f.write('Before pruning: Adv loss: %.8f, Adv Acc: %.3f\n' % (advloss, 1. * advcorr/len(test_loader.dataset)))
    f.write('=============== Start Pruning ==================\n')

    for epoch in range(1, args.epochs+1):
        f.write('Pruning Iteration {}\n'.format(epoch))
        
        if args.prune_method == 'global':
            total, pruned = global_threshold_prune(model, device, epoch, args)
        elif args.prune_method == 'layerwise':
            total, pruned = layerwise_threshold_prune(model, device, epoch, args)

        if args.opt == 'adam':
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        else:
            raise ValueError('Optimizer {} is not supported. Try SGD (with Momentum) and Adam instead'.format(args.opt))

        #f.write('Pruning Iteration {}\n'.format(epoch))
        #f.write('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}\n'.format(total, pruned, pruned/total))
        
        clnloss, clncorr, advloss, advcorr = test(model, test_loader, adversary, device, args)
        f.write('Before retraining: Cln loss: %.8f, Cln Acc: %.3f\n' % (clnloss, 1. * clncorr/len(test_loader.dataset)))
        f.write('Before retraining: Adv loss: %.8f, Adv Acc: %.3f\n' % (advloss, 1. * advcorr/len(test_loader.dataset))) 
        for retrain_epoch in range(1, args.retrain_epochs+1):
            lr = args.lr
            for param_group in opt.param_groups:
                param_group['lr'] = lr
                print("current learning rate: {}".format(lr))

            finetune(model, train_loader, opt, retrain_epoch, adversary, device, args)
            if retrain_epoch % 5 == 0:
                test(model, test_loader, adversary, device, args)

        torch.save(model.state_dict(), '{}_{}.pt'.format(args.save_model, epoch))
        clnloss, clncorr, advloss, advcorr = test(model, test_loader, adversary, device, args)
        f.write('After retraining: Cln loss: %.8f, Cln Acc: %.3f\n' % (clnloss, 1. * clncorr/len(test_loader.dataset)))
        f.write('After retraining: Adv loss: %.8f, Adv Acc: %.3f\n' % (advloss, 1. * advcorr/len(test_loader.dataset))) 
        f.write('==============================================\n')

        print("Pruning Sanity Check")
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float()
                print("Total param in layer{}: {}, Non-zero {}".format(k, mask.numel(), int(torch.sum(mask))))

    f.close()
        
    
if __name__ == '__main__':
    main()


# In[ ]:




