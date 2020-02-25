import os
import argparse
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

import copy

from adversarial_training_utils import *
from model_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--log_interval', default=200, type=int)

    parser.add_argument('--disable_adv', action='store_true', default=False)
    parser.add_argument('--no_adv_eval', action='store_true', default=False)
    parser.add_argument('--attack_type', default='pgd', type=str)
    parser.add_argument('--eps', default=8.0/255.0, type=float)
    parser.add_argument('--nb_iter', default=7, type=int)
    parser.add_argument('--eps_iter', default=2.0/255.0, type=float)
    parser.add_argument('--nb_random_start', default=1, type=int)
    parser.add_argument('--l2pgd_init_method', default='rand', type=str)

    parser.add_argument('--model_type', default='wrn', type=str, help='wrn, vgg')
    parser.add_argument('--scale_factor', default=1, type=float)
    parser.add_argument('--vgg_depth', default=11, type=int)
    parser.add_argument('--vgg_batch_norm', default=False, action='store_true')
  
    parser.add_argument('--transfer_from', type=str)
    parser.add_argument('--transfer_model_type', default='wrn', type=str, help='wrn, vgg')
    parser.add_argument('--transfer_scale_factor', default=1, type=float)
    parser.add_argument('--transfer_vgg_depth', default=11, type=int)
    parser.add_argument('--transfer_vgg_batch_norm', default=False, action='store_true')

    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--adam_beta2', default=0.99, type=float)
    parser.add_argument('--early_stop', default=False, action='store_true')
    parser.add_argument('--early_stop_criteria', default='adv', type=str)

    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--weight_decay_schedule', default=False, action='store_true')
    parser.add_argument('--weight_decay_schedule_method', default='undecay', type=str)
    parser.add_argument('--weight_decay_undecay_rate', default=10, type=int)
    parser.add_argument('--spectral_norm', action='store_true', default=False)
    parser.add_argument('--spectral_iter', default=1, type=int)
    parser.add_argument('--residual_spectral_norm', action='store_true', default=False)
    parser.add_argument('--orthonormal', default=0.0, type=float)
    parser.add_argument('--orth_ord', default='fro', type=str)
    parser.add_argument('--orth_double', action='store_true', default=False)
    parser.add_argument('--jacobian', default=0, type=float)
    parser.add_argument('--jacobian_ord', default=2, type=int)

    parser.add_argument('--augment', default=False, action='store_true')

    parser.add_argument('--save_initialization', type=str)
    parser.add_argument('--save_model', default='cifar_adv_train.pt', type=str)
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--logs', type=str)

    args = parser.parse_args()

    if args.orth_ord != 'fro':
        args.orth_ord = int(args.orth_ord) if not args.orth_ord == 'inf' else float(args.orth_ord)
    torch.manual_seed(args.seed)
    
    # For debugging at local desktop without GPU
    # device = torch.device("cpu")
    # If GPU is available:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    data_dir = '../data/tiny-imagenet-200'
    train_transforms = transforms.Compose([transforms.RandomCrop(64, padding=8),\
                                           transforms.RandomHorizontalFlip(),\
                                           transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=100)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)

    model = get_models(args.model_type, args.load_model, args.scale_factor, device, dataset='tinyimagenet', \
                       batch_norm=args.vgg_batch_norm, tinyimagenet_vgg_depth=args.vgg_depth)
    print(model)
    if args.save_initialization is not None:
        torch.save(model.state_dict(), args.save_initialization)

    if args.optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.adam_beta2), weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
    
    max_correct = 0
    early_stop_epoch = 0

    max_eps = copy.deepcopy(args.eps)
    if args.transfer_from is not None:
        transfer = get_models(args.transfer_model_type, args.transfer_from, args.transfer_scale_factor, device, \
                              dataset='tinyimagenet', batch_norm=args.transfer_vgg_batch_norm, \
                              tinyimagenet_vgg_depth=args.transfer_vgg_depth) 
        adversary = get_adversary(args.attack_type, transfer, args)
    else:
        adversary = get_adversary(args.attack_type, model, args)

    f = open(args.logs, 'w')
    f.write('Training conditions\n')
    f.write('Adversarial Training? {}\n'.format(not(args.disable_adv)))
    f.write('Adversary Type: {} iter {} eps {}\n'.format(args.attack_type, args.nb_iter, args.eps))
    f.write('Adversary transferred? {}\n'.format(True if args.transfer_from is not None else False))
    f.write('Training Model: {} scaled to {}\n'.format(args.model_type, args.scale_factor))
    f.write('Train optimizer {} with initial lr {} decayed by 0.1 every 40 epochs\n'.format(args.optimizer, args.lr))
    f.write('Trained to max epochs {}, Early stopping? {} (criteria: {})\n'.format(args.epochs, args.early_stop, args.early_stop_criteria))
    f.write('\n Start Training...\n')
    f.write('epoch\tclnloss\tclncorr\tadvloss\tadvcorr\n')

    for epoch in range(1, args.epochs+1):
        lr = max(args.lr * (0.1 ** (epoch // 40)), args.lr * (0.1 ** 3))
        for param_group in opt.param_groups:
            param_group['lr'] = lr
            print('Learning rate for current iteration: {}'.format(lr))
            if args.weight_decay_schedule:
                if args.weight_decay_schedule_method == 'boost':
                    weight_decay = args.weight_decay if epoch < 90 else 1e-2
                elif args.weight_decay_schedule_method == 'undecay':
                    weight_decay = args.weight_decay * (args.weight_decay_undecay_rate ** (epoch // 40))
                print('Weight decay for current iteration: {}'.format(weight_decay))
                param_group['weight_decay'] = weight_decay
        train(model, train_loader, opt, epoch, adversary, device, args)
        clnloss, clncorr, advloss, advcorr = test(model, val_loader, adversary, device, args)
        f.write('%d\t%.8f\t%d\t%.8f\t%d\n'%(epoch, clnloss, clncorr, advloss, advcorr))
        if args.early_stop:
            criteria = advcorr if (args.early_stop_criteria == 'adv') else clncorr
            if criteria > max_correct:
                max_correct = criteria
                early_stop_epoch = epoch
                torch.save(model.state_dict(), '{}_{}.pt'.format(args.save_model, 'early_stop'))

    torch.save(model.state_dict(), '{}_{}.pt'.format(args.save_model, 'full_train'))
    print("Early Stop Epoch @ {}".format(early_stop_epoch))
    f.write('Train ended!\n')
    f.write('Early Stop Epoch @ {}'.format(early_stop_epoch))
    f.close()
    
if __name__ == '__main__':
    main()


# In[ ]:




