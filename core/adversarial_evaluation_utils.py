# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

# Add path to advertorch folder
import sys
sys.path.insert(0, './compensated_attacks')
sys.path.insert(0, './core')

from one_step_gradient import FGSM, FGM, RFGSM, RFGM
from iterative_projected_gradient import LinfPGDAttack, L2PGDAttack

# Model classes for bpda substitute models
from bpda_models import *
from tinyimagenet_bpda_models import *

# Python program to print 
# colored text and background 
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) 
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk)) 
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk)) 
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk)) 

def generate_adversarial_temperature_softmax(data, target, adversary, temperature=100.):
    # input: data, target must be correctly located at desired device (cpu or cuda)
    # adversary in advertorch's attacks; except for custom-defined modified attacks
    # returns perturbed adversarial data
    
    ##### TODO #####
    # Check if given adversary has temperature softmax property
    # if not, raise error
    ################
    
    advdata = adversary.perturb(data.clone(), target.clone(), True, temperature)
    return advdata

def generate_adversarial_targeted_objective(data, target, adversary, output, targeted_to='second'):
    # set the adversary's 'targeted' to True. before returning, reset the adversary's 'targeted' to False
    # define the targeted objective with input 'targeted_to'
    # output: original logits on clean data - necessary for least-likely and second
    
    if targeted_to == 'random':
        targeted_obj = torch.randint(0, 10, size=target.size(), dtype=target.dtype, device=target.device)
    elif targeted_to == 'least-likely':
        targeted_obj = output.min(1, keepdim=True)[1].view_as(target)
    elif targeted_to == 'second':
        targeted_obj = output.topk(2, dim=1)[1].t()[1].view_as(target)
    else:
        raise TypeError('Targeted objective to random, least-likely, or second.')
    
    targeted = copy.deepcopy(adversary.targeted)
    adversary.targeted = True
    advdata = adversary.perturb(data.clone(), targeted_obj.clone())
    adversary.targeted = targeted
    return advdata
    
def fetch_bpda_submodel(model, adversary, relu_replace='softplus', relu_replace_slope_param=2.0, relu_replace_threshold=2.0,\
                        max_pool_replace_param=5, model_type='simple', model_scale_factor=1., \
                        dataset='cifar', tinyimagenet_vgg_depth=11, tinyimagenet_vgg_batch_norm=False):
    # Fetch bpda submodel and corresponding adversary based on original model and adversary
    if dataset == 'cifar' or dataset == 'svhn':
        if model_type == 'simple':
            sub_model = bpda_wrapped_simple(model, model_scale_factor, spectral_norm=model.spectral_norm, spectral_iter=1,\
                                            batch_norm=model.batch_norm, p=max_pool_replace_param, relu=relu_replace,\
                                            relu_slope=relu_replace_slope_param, softplus_threshold=relu_replace_threshold)
        elif model_type == 'wrn':
            sub_model = bpda_wrapped_wrn(model, depth=28, num_classes=10, widen_factor=model_scale_factor,\
                                         spectral_norm=model.spectral_norm, spectral_iter=1, res_spec=False,\
                                         relu_replace=relu_replace, relu_replace_slope_param=relu_replace_slope_param,\
                                         relu_replace_threshold=relu_replace_threshold)
        else:
            raise TypeError('Only supports model types: simple (both no BN/BN) and wrn')
    elif dataset == 'tinyimagenet':
        if model_type == 'wrn':
            sub_model = bpda_wrapped_tinyimagenet_wrn(model, model_scale_factor, relu_replace, \
                                                      relu_replace_slope_param, relu_replace_threshold)
        elif model_type == 'vgg':
            sub_model = bpda_wrapped_vgg(model, model_scale_factor, tinyimagenet_vgg_depth, tinyimagenet_vgg_batch_norm,\
                                         relu_replace, relu_replace_slope_param, relu_replace_threshold, max_pool_replace_param)
        else:
            raise TypeError('Only supports model types: VGG and WRN-50')
    
    sub_model.eval()
    sub_adversary = copy.deepcopy(adversary)
    sub_adversary.predict = sub_model
    return sub_model, sub_adversary

def generate_adversarial_zero_loss_compensation(data, target, adversary, output, zero_loss_compensation='none',\
                                                temperature=100., targeted_to='second'):
    if zero_loss_compensation == 'none':
        advdata = adversary.perturb(data.clone(), target.clone())
        return advdata
    elif zero_loss_compensation == 'temperature':
        return generate_adversarial_temperature_softmax(data, target, adversary, temperature)
    elif zero_loss_compensation == 'targeted':
        return generate_adversarial_targeted_objective(data, target, adversary, output, targeted_to)
    
def generate_adversarial_second_order_init(data, target, adversary, output, init_method='miyato',\
                                           zero_loss_compensation='none', bpda_compensation=False, \
                                           temperature=100., targeted_to='second'):
    # From the adversary (iterative types; PGD), change the initialization method
    original_init_method = copy.deepcopy(adversary.init_method)
    adversary.init_method = init_method
    # If bpda_compensation is true, then adversary should be **sub_adversary** for BPDA-wrapped model
#     if bpda_compensation:
#         print("Your BPDA compensation is set ON. Double-check if your adversary is sub-adversary with BPDA-wrapped functions")
    advdata = generate_adversarial_zero_loss_compensation(data, target, adversary, output, zero_loss_compensation,\
                                                          temperature, targeted_to)
    adversary.init_method = original_init_method
    return advdata
            
def evaluate_all(model, loader, adversary, device, \
                 zero_loss_compensation='none', temperature=100., targeted_to='random',
                 bpda_compensation=False, relu_replace='softplus', relu_replace_slope_param=2.0, relu_replace_threshold=2.0,\
                 max_pool_replace_param=5,\
                 iterative=False, second_order_init_method='rand',\
                 model_type='simple', model_scale_factor=1.,\
                 transfer=None,\
                 dataset='cifar', tinyimagenet_vgg_depth=11, tinyimagenet_vgg_batch_norm=False):
    # We evaluate all possible combinations of modifications given certain type of adversary
    # Report following accuracies:
    # - Clean: conventional 'accuracy'
    # - Vanilla adversarial: conventional empirical evaluation for adversarial robustness
    # - Zero-loss compensated adversarial only: zero-loss compensation method given in the input argument
    # - Non-differentiability compensated adversarial only (bpda): with function replacement stated in argument
    # - Zero-loss + Non-differentiability compensated adversarial
    # - (Iterative attacks only) Second-order initialization method only: as stated in argument
    # - (Iterative attacks only) Second-order initialization + zero-loss
    # - (Iterative attacks only) Second-order initialization + zero-loss + non-differentiability
    clean_acc = 0                   # Clean accuracy
    vanilla_adv_acc = 0             # Vanilla adversarial accuracy
    zero_only_acc = 0               # Apply zero-compensation for those survivied vanilla attack
    if bpda_compensation:
        nondiff_only_acc = 0        # Apply non-diff copmensation for those survived vanilla attack
        zero_nondiff_acc = 0        # Apply both zero-compensation and non-diff compensation for those survivied vanilla attack
    if iterative:
        second_only_acc = 0         # Apply second-order init method for iterative attacks for those survivied vanilla
        second_zero_acc = 0         # Apply both zero-compensation and second-order init for those survivied vanilla
        if bpda_compensation:
            second_zero_nondiff_acc = 0 # Apply zero-compensation, non-diff compensation, and second-order init method 
        
    prCyan("Evaluation scheme")
    prGreen("Adversary type: {}".format(type(adversary)))
    prGreen("Zero-loss compensation: {} ({})".format(zero_loss_compensation,\
                                                   temperature if zero_loss_compensation=='temperature' else\
                                                   (targeted_to if zero_loss_compensation=='targeted' else '-')))
    prGreen("BPDA compensation: {}".format(bpda_compensation))
    if iterative:
        prGreen("Second-order initialization method: {}".format(second_order_init_method))
    
    model.eval()
    if transfer is not None:
        transfer.eval()
    if bpda_compensation:
        if transfer is not None:
            sub_model, sub_adversary = fetch_bpda_submodel(transfer, adversary, relu_replace, relu_replace_slope_param,\
                                                           relu_replace_threshold, max_pool_replace_param, model_type, \
                                                           model_scale_factor, dataset, \
                                                           tinyimagenet_vgg_depth, tinyimagenet_vgg_batch_norm)
        else:
            sub_model, sub_adversary = fetch_bpda_submodel(model, adversary, relu_replace, relu_replace_slope_param,\
                                                           relu_replace_threshold, max_pool_replace_param, model_type, \
                                                           model_scale_factor, dataset, \
                                                           tinyimagenet_vgg_depth, tinyimagenet_vgg_batch_norm)
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        # First, evaluate clean accuracy
        with torch.no_grad():
            cln_output = model(data)
        pred = cln_output.max(1, keepdim=True)[1]
        clean_acc += pred.eq(target.view_as(pred)).sum()
        
        # Vanilla adversarial accuracy
        advdata = adversary.perturb(data.clone(), target.clone())
        with torch.no_grad():
            output = model(advdata)
        pred = output.max(1, keepdim=True)[1]
        vanilla_adv_acc += pred.eq(target.view_as(pred)).sum()
        
        # Mask those survived vanilla adversary
        mask = pred.eq(target.view_as(pred)).view(-1).nonzero().view(-1)
        if mask.numel() > 0:
            masked_data = torch.index_select(data, 0, mask)
            masked_target = torch.index_select(target, 0, mask)
            masked_output = torch.index_select(cln_output, 0, mask)
        
            # Zero-loss compensation only
            # If we are transferring adversarial examples, i.e. for black-box,
            # we assume we don't have access to logits of the target model
            # Thus, masked_output fed into generate_adversarial_zero_loss_compensation has to be 
            # the logit of surrogate model (transfer)
            if transfer is not None:
                with torch.no_grad():
                    tr_output = transfer(data)
                masked_output = torch.index_select(tr_output, 0, mask) # mask should be the same!
            advdata = generate_adversarial_zero_loss_compensation(masked_data, masked_target, adversary, masked_output, \
                                                                  zero_loss_compensation, temperature, targeted_to)
            with torch.no_grad():
                output = model(advdata)
            pred = output.max(1, keepdim=True)[1]
            zero_only_acc += pred.eq(masked_target.view_as(pred)).sum()

            if bpda_compensation:
                # BPDA-only
                #advdata = generate_adversarial_zero_loss_compensation(masked_data, masked_target, sub_adversary, masked_output)
                advdata = sub_adversary.perturb(masked_data.clone(), masked_target.clone())
                with torch.no_grad():
                    output = model(advdata)
                pred = output.max(1, keepdim=True)[1]
                nondiff_only_acc += pred.eq(masked_target.view_as(pred)).sum()

                # BPDA+Zero: 1) mask surviving data after bpda
                #            2) try zero-compensation
                #            3) and zero-compensation + bpda: report # data survived both 2 and 3
                mask2 = pred.eq(masked_target.view_as(pred)).view(-1).nonzero().view(-1)
                if mask2.numel() > 0:
                    masked_data_2 = torch.index_select(masked_data, 0, mask2)
                    masked_target_2 = torch.index_select(masked_target, 0, mask2)
                    masked_output_2 = torch.index_select(masked_output, 0, mask2)
                    advdata = generate_adversarial_zero_loss_compensation(masked_data_2, masked_target_2, adversary, masked_output_2,\
                                                                          zero_loss_compensation, temperature, targeted_to)
                    with torch.no_grad():
                        output1 = model(advdata)
                    pred1 = output1.max(1, keepdim=True)[1]

                    advdata = generate_adversarial_zero_loss_compensation(masked_data_2, masked_target_2, sub_adversary,\
                                                                          masked_output_2, zero_loss_compensation, temperature,\
                                                                          targeted_to)
                    with torch.no_grad():
                        output2 = model(advdata)
                    pred2 = output2.max(1, keepdim=True)[1]

                    zero_nondiff_acc += (pred1.eq(masked_target_2.view_as(pred1)) & pred2.eq(masked_target_2.view_as(pred2))).sum()

            # For iterative attacks
            if iterative:
                # second-init only
                advdata = generate_adversarial_second_order_init(masked_data, masked_target, adversary, masked_output,
                                                                 init_method=second_order_init_method)
                with torch.no_grad():
                    output = model(advdata)
                pred = output.max(1, keepdim=True)[1]
                second_only_acc += pred.eq(masked_target.view_as(pred)).sum()

                # Zero + second
                # 1) mask the data that survived above
                # 2) Try zero+second 
                # 3) measure # data survived 2
                # Zero + second + bpda
                # on those survivied until 3, apply bpda (bpda+second, bpda+second+zero)
                mask3 = pred.eq(masked_target.view_as(pred)).view(-1).nonzero().view(-1)
                if mask3.numel() > 0:
                    masked_data_3 = torch.index_select(masked_data, 0, mask3)
                    masked_target_3 = torch.index_select(masked_target, 0, mask3)
                    masked_output_3 = torch.index_select(masked_output, 0, mask3)
                    
                    # zero-compensation
                    advdata = generate_adversarial_zero_loss_compensation(masked_data_3, masked_target_3, adversary, \
                                                                          masked_output_3, \
                                                                          zero_loss_compensation, temperature, targeted_to)
                    with torch.no_grad():
                        output1 = model(advdata)
                    pred1 = output1.max(1, keepdim=True)[1]
                    
                    # zero + second
                    advdata = generate_adversarial_second_order_init(masked_data_3, masked_target_3, adversary, masked_output_3,\
                                                                     second_order_init_method, zero_loss_compensation,\
                                                                     temperature=temperature, targeted_to=targeted_to)
                    with torch.no_grad():
                        output2 = model(advdata)
                    pred2 = output2.max(1, keepdim=True)[1]
                    
                    pred12 = (pred1.eq(masked_target_3.view_as(pred1)) & pred2.eq(masked_target_3.view_as(pred2)))
                    second_zero_acc += pred12.sum()
                    
                    if bpda_compensation:
                        # Try bpda + second + zero
                        advdata = generate_adversarial_second_order_init(masked_data_3, masked_target_3, sub_adversary, \
                                                                         masked_output_3, second_order_init_method,\
                                                                         zero_loss_compensation, bpda_compensation, temperature,\
                                                                         targeted_to)
                        with torch.no_grad():
                            output3 = model(advdata)
                        pred3 = output3.max(1, keepdim=True)[1]
                        
                        second_zero_nondiff_acc += (pred12 & pred3.eq(masked_target_3.view_as(pred3))).sum()
                        
    # Print the result
    num_data = float(len(loader.dataset))
    prYellow("Clean accuracy: {}/{} ({:.4f}%)".format(clean_acc, num_data, 100.*float(clean_acc)/num_data))
    prYellow("Vanilla adversarial accuracy: {}/{} ({:.4f}%)".format(vanilla_adv_acc, num_data, 100.*float(vanilla_adv_acc)/num_data))
    prPurple("Zero-compensated adversarial accuracy: {}/{} ({:.4f}%)"\
             .format(zero_only_acc, num_data, 100.*float(zero_only_acc)/num_data))
    if bpda_compensation:
        prPurple("Nondiff-compensated adversarial (BPDA) accuracy: {}/{} ({:.4f}%)".format(nondiff_only_acc, num_data,\
                                                                                       100.*float(nondiff_only_acc)/num_data))
        prPurple("Nondiff+Zero-compensated adversarial accuracy: {}/{} ({:.4f}%)".format(zero_nondiff_acc, num_data,\
                                                                                     100.*float(zero_nondiff_acc)/num_data))
    if iterative:
        prPurple("Second-order-initialization adversarial accuracy: {}/{} ({:.4f}%)".format(second_only_acc, num_data,\
                                                                                        100.*float(second_only_acc)/num_data))
        prPurple("Second-order-initialization + Zero-compensated adversarial accuracy: {}/{} ({:.4f}%)".\
                 format(second_zero_acc, num_data, 100.*float(second_zero_acc)/num_data))
        if bpda_compensation:
            prPurple("Second-order-initialization + Nondiff+Zero-compensated adversarial accuracy: {}/{} ({:.4f}%)"\
                  .format(second_zero_nondiff_acc, num_data, 100.*float(second_zero_nondiff_acc)/num_data))
    return

def get_adversary(model, attack_type, eps, nb_iter=1, eps_iter=1., nb_random_start=1.):
    if attack_type == 'pgd':
        adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction='sum'), eps=eps, nb_iter=nb_iter,\
                                  eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0)
        iterative = True
    elif attack_type == 'l2pgd':
        adversary = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction='sum'), eps=eps, nb_iter=nb_iter, \
                                eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0)
        iterative = True
    elif attack_type == 'fgsm':
        adversary = FGSM(model, loss_fn=nn.CrossEntropyLoss(reduction='sum'), eps=eps, clip_min=0.0, clip_max=1.0)
        iterative = False
    elif attack_type == 'fgm':
        adversary = FGM(model, loss_fn=nn.CrossEntropyLoss(reduction='sum'), eps=eps, clip_min=0.0, clip_max=1.0)
        iterative = False
    elif attack_type == 'rfgsm':
        adversary = RFGSM(model, loss_fn=nn.CrossEntropyLoss(reduction='sum'), eps=eps, clip_min=0.0, clip_max=1.0)
        iterative = False
    elif attack_type == 'rfgm':
        adversary = RFGM(model, loss_fn=nn.CrossEntropyLoss(reduction='sum'), eps=eps, clip_min=0.0, clip_max=1.0)
        iterative = False
    else:
        raise TypeError('Not supported adversary type')
    return adversary, iterative
    