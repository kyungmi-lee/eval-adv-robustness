import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#from torchvision import datasets, transforms

# Add path to advertorch folder
import sys
sys.path.insert(0, './advertorch')
from advertorch.context import ctx_noparamgrad_and_eval

def finetune(model, train_loader, opt, epoch, adversary, device, args):
    model.train()
    #print(model)
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
        loss = criterion(output, target)
        
        loss.backward()
        #if args.prune:
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and args.prune_all):
                #print(m.weight.grad)
                #import copy
                #print(m, m.weight.grad)
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data *= mask
                
                #if isinstance(m, nn.BatchNorm2d):
                #    m.bias.grad.data *= mask

        opt.step()

        if batch_idx % args.log_interval == 0:
        
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset), 
            100.* batch_idx / len(train_loader), loss.item()))

def get_mask(pruned_model, args):
    mask = []
    for k, m in enumerate(pruned_model.modules()):
        if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and args.prune_all):
            zero_mask = m.weight.data.abs().clone().gt(0).float()
            mask.append(zero_mask)
    return mask

def global_threshold_prune(model, device, epoch, args):
    total = 0
    for m in model.modules():
        # Only prune the conv layer | prune linear as well if args.prune_all=True
        if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and args.prune_all):
            total += m.weight.data.numel()
    conv_weights = torch.zeros(total, device=device)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and args.prune_all):
            size = m.weight.data.numel()
            conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    if args.prune_fixed_percent:
        prune_ratio = 1 - (args.prune_ratio_per_step ** epoch)
        thre_index = int(total * prune_ratio)
    else:
        thre_index = int(total * epoch * args.prune_ratio / args.epochs)
        #if args.prune_all and (epoch * args.prune_ratio / args.epochs > 0.9):
        #thre_index = int(total * 0.95 / args.epochs)
    thre = y[thre_index]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))

    zero_flag = False
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and args.prune_all):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask))==0:
                zero_flag = True
                print('Warning: There is a layer with all zeros')
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.format(k, mask.numel(), int(torch.sum(mask))))

    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    return total, pruned
            
def layerwise_threshold_prune(model, device, epoch, args):
    total = 0
    num_layer = 0

    thre = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and args.prune_all):
            size = m.weight.data.numel()
            conv_weights = m.weight.data.view(-1).abs().clone()
            y, i = torch.sort(conv_weights)
            if args.prune_fixed_percent:
                prune_ratio = 1 - (args.prune_ratio_per_step ** epoch)
                thre_index = int(size * prune_ratio)
            else:
                thre_index = int(size * epoch * args.prune_ratio / args.epochs)
            thre.append(y[thre_index])
            total += size
            num_layer += 1
 
    pruned = 0
    for i in range(len(thre)):
        print('Pruning threshold for layer {}: {}'.format(i, thre[i]))

    zero_flag = False
    num_layer = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and args.prune_all):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre[num_layer]).float()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            num_layer += 1
            if int(torch.sum(mask))==0:
                zero_flag = True
                print('Warning: There is a layer with all zeros')
                #f.write('Warning: There is a layer with all zeros\n')
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.format(k, mask.numel(), int(torch.sum(mask))))
            #f.write('layer index: {:d} \t total params: {:d} \t remaining params: {:d}\n'.format(k, mask.numel(), int(torch.sum(mask))))
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    return total, pruned




