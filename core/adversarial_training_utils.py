#import matplotlib.pyplot as plt
import os
import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#from torchvision import datasets, transforms

# Add path to advertorch folder
import sys
sys.path.insert(0, './advertorch')
sys.path.insert(0, './compensated_attacks')
sys.path.insert(0, './core')

from advertorch.attacks import CarliniWagnerL2Attack
from advertorch.context import ctx_noparamgrad_and_eval

from one_step_gradient import RFGSM, RFGM, FGSM, FGM
from iterative_projected_gradient import LinfPGDAttack, L2PGDAttack

from orthonormal_regularization import orthonormal_regularization as orth
from jacobian_regularization import jacobian_regularization as jaco

def train(model, train_loader, opt, epoch, adversary, device, args):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # generate adversarial sample
        if not args.disable_adv:
            with ctx_noparamgrad_and_eval(model):
                data_adv = adversary.perturb(data, target)
                
        opt.zero_grad()
        if args.disable_adv:
            output = model(data)
        else:
            output = model(data_adv)
            if args.augment:
                output = torch.cat([output, model(data)], dim=0)
        if args.augment:
            loss = criterion(output, torch.cat([target, target], dim=0))
        else:
            loss = criterion(output, target)
            
        if args.orthonormal > 0:
            orth_loss = args.orthonormal * orth(model, args.orth_ord, args.orth_double)
            loss += orth_loss
            #loss += args.orthonormal * orth(model)
        if hasattr(args, 'jacobian') and args.jacobian > 0:
            jaco_loss = args.jacobian * jaco(model, data, target, criterion, p=args.jacobian_ord)
            loss += jaco_loss
            
        loss.backward()
        opt.step()
       
        if hasattr(args, 'train_val_split') and args.train_val_split:
            loader_len = len(train_loader.sampler.indices)
        else:
            loader_len = len(train_loader.dataset)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), loader_len, 
            100.* float(batch_idx) * len(data) / float(loader_len), loss.item()))
            if args.orthonormal > 0:
                print(orth_loss.item())
            if hasattr(args, 'jacobian') and args.jacobian > 0:
                print(jaco_loss.item())

def test(model, test_loader, adversary, device, args):
    model.eval()
    test_clnloss = 0
    clncorrect = 0
    
    test_advloss = 0
    advcorrect= 0
   
    criterion = nn.CrossEntropyLoss()
    for clndata, target in test_loader:
        clndata, target = clndata.to(device), target.to(device)
        with torch.no_grad():
            output = model(clndata)
        test_clnloss += criterion(output, target).item()
        pred = output.max(1, keepdim=True)[1]
        clncorrect += pred.eq(target.view_as(pred)).sum().item()
        
        if not args.no_adv_eval:
            _target = copy.deepcopy(target)
            random_start_advcorrect = torch.zeros(target.size()[0], args.nb_random_start)
            for i in range(args.nb_random_start):
                advdata = adversary.perturb(clndata, target)
                if args.attack_type == 'cwl2':
                    # Measure the distance
                    dist = torch.norm(advdata.view(clndata.shape[0], -1)-clndata.view(clndata.shape[0], -1), p=2, dim=1)
                    mask = torch.gt(dist, args.eps)
                elif args.attack_type == 'cwlinf':
                    dist = torch.norm(advdata.view(clndata.shape[0], -1)-clndata.view(clndata.shape[0], -1), p=float('inf'), dim=1)
                    mask = torch.gt(dist, args.eps)
                    
                with torch.no_grad():
                    adv_output = model(advdata)
                test_advloss += criterion(adv_output, _target).item() / args.nb_random_start
                pred = adv_output.max(1, keepdim=True)[1]
                random_start_advcorrect[:, i] = pred.eq(_target.view_as(pred)).squeeze()
                if args.attack_type == 'cwl2' or args.attack_type == 'cwlinf':
                    # For the points where the distance under the given norm exceeds the pre-defined epsilon,
                    # note attacks on those points fail (Only for C&W attack - where we cannot predefine epsilon)
                    # Thus, make sure the results corresponding to the mask (points greater than the epsilon) to be 'True'
                    if args.cw_invalidate_dist_larger_than_eps:
                        random_start_advcorrect[:][mask] = 1
            advcorrect += (random_start_advcorrect.sum(dim=1)/args.nb_random_start).int().sum().item()
    
    if hasattr(args, 'train_val_split') and args.train_val_split:
        loader_len = len(test_loader.sampler.indices)
    else:
        loader_len = len(test_loader.dataset)

    test_clnloss /= len(test_loader.dataset)
    print('\nClean set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_clnloss, clncorrect, loader_len,
        100. * clncorrect / loader_len))
    

    if not args.no_adv_eval:
        test_advloss /= len(test_loader.dataset)
        print('\nAdv set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_advloss, advcorrect, loader_len,
            100. * advcorrect / loader_len))

        return (test_clnloss, clncorrect, test_advloss, advcorrect)
    return (test_clnloss, clncorrect, 0, 0)

# In[ ]:
def get_adversary(attack_type, attack_on, args):
    if hasattr(args, 'targeted'):
        _targeted = args.targeted
    else:
        _targeted = False
        
    if attack_type == 'pgd':
        adversary = LinfPGDAttack(attack_on, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.eps, nb_iter=args.nb_iter, eps_iter=args.eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=_targeted)
    elif attack_type == 'fgsm':
        adversary = FGSM(attack_on, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.eps, clip_min=0., clip_max=1., targeted=_targeted)
    elif attack_type == 'rfgsm':
        adversary = RFGSM(attack_on, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.eps, clip_min=0., clip_max=1., targeted=_targeted)
    elif attack_type == 'l2pgd':
        adversary = L2PGDAttack(attack_on, eps=args.eps, nb_iter=args.nb_iter, eps_iter=args.eps_iter, rand_init=True, clip_min=0.0,\
                                clip_max=1.0, targeted=_targeted)
    elif attack_type == 'fgm':
        adversary = FGM(attack_on, loss_fn=nn.CrossEntropyLoss(reduction='sum'), eps=args.eps, clip_min=0., clip_max=1.,\
                        targeted=_targeted)
    elif attack_type == 'rfgm':
        adversary = RFGM(attack_on, loss_fn=nn.CrossEntropyLoss(reduction='sum'), eps=args.eps, clip_min=0., clip_max=1.,\
                        targeted=_targeted)
    elif attack_type == 'cwl2':
        adversary = CarliniWagnerL2Attack(attack_on, 10, max_iterations=1000) #Typical. For more precise results, use larger iterations
    else:
        print("Attack type {} is not supported. Please use either PGD or FGSM".format(attack_type))
        adversary = None
        
    return adversary

