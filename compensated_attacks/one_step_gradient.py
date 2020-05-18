from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch

import sys
sys.path.insert(0, './advertorch')
sys.path.insert(0, './compensated_attacks')

from advertorch.utils import batch_multiply
from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin

# Baseline attacks implemented at advertorch
from advertorch.attacks import FGSM as FGSM_base
from advertorch.attacks import FGM  as FGM_base

from utils import *

# R-FGSM and R-FGM implemented using basic attack class and operations provided by AdverTorch
# FGSM and FGM modified for logit rescaling option

class RandomGradientSignAttack(Attack, LabelMixin):

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False):
        
        super(RandomGradientSignAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None, temperature_softmax=False, temperature_softmax_value=None):

        x, y = self._verify_and_process_inputs(x, y)
        delta = torch.randn(x.size(), device=x.device).sign()
        delta = nn.Parameter(delta)
        
        if isinstance(self.eps, float):
            delta.data = batch_multiply(self.eps/2, delta.data)
        else:
            delta.data = channel_eps_multiply(delta.data, self.eps, scale=0.5)
            
        if isinstance(self.clip_min, float):
            delta.data = clamp(x.data+delta.data, min=self.clip_min, max=self.clip_max) - x.data
        else:
            delta.data = channel_clip(x.data+delta.data, self.clip_min, self.clip_max) - x.data
        delta = delta.requires_grad_()

        #xadv = x.requires_grad_()
        outputs = self.predict(x + delta)
        if temperature_softmax:
            if temperature_softmax_value:
                outputs = outputs / temperature_softmax_value
            else:
                outputs = outputs / torch.std(outputs.clone().detach(), dim=1, keepdim=True)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad_sign = delta.grad.detach().sign()
        
        if isinstance(self.eps, float):
            delta.data = delta.data + batch_multiply(self.eps/2, grad_sign)
        else:
            delta.data = channel_eps_multiply(grad_sign, self.eps, scale=0.5) + delta.data
            
        if isinstance(self.clip_min, float):
            delta.data = clamp(x.data+delta.data, self.clip_min, self.clip_max) - x.data
        else:
            delta.data = channel_clip(x.data+delta.data, self.clip_min, self.clip_max) - x.data

        #xadv = xadv + self.eps * grad_sign
        #xadv = clamp(xadv, self.clip_min, self.clip_max)
        xadv = x + delta

        return xadv


RFGSM = RandomGradientSignAttack

class GradientSignAttack(FGSM_base):
    
    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0., clip_max=1., targeted=False):
        super(GradientSignAttack, self).__init__(predict, loss_fn, eps, clip_min, clip_max, targeted)
        
    def perturb(self, x, y=None, temperature_softmax=False, temperature_softmax_value=None):
        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)
        if temperature_softmax:
            if temperature_softmax_value:
                outputs = outputs / temperature_softmax_value
            else:
                outputs = outputs / torch.std(outputs.clone().detach(), dim=1, keepdim=True)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad_sign = xadv.grad.detach().sign()
        # print(torch.max(grad_sign), torch.min(grad_sign))
        
        if isinstance(self.eps, float):
            xadv = xadv + self.eps * grad_sign
        else:
            xadv = xadv + channel_eps_multiply(grad_sign, self.eps)
            
        if isinstance(self.clip_min, float):
            xadv = clamp(xadv, self.clip_min, self.clip_max)
        else:
            xadv = channel_clip(xadv, self.clip_min, self.clip_max)

        return xadv


FGSM = GradientSignAttack

class GradientAttack(FGM_base):
    
    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0., clip_max=1., targeted=False):
        if not isinstance(eps, float):
            eps = (eps[0] + eps[1] + eps[2]) / 3.
        super(GradientAttack, self).__init__(predict, loss_fn, eps, clip_min, clip_max, targeted)
        
    def perturb(self, x, y=None, temperature_softmax=False, temperature_softmax_value=None):
        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)
        if temperature_softmax:
            if temperature_softmax_value:
                outputs = outputs / temperature_softmax_value
            else:
                outputs = outputs / torch.std(outputs.clone().detach(), dim=1, keepdim=True)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad = normalize_by_pnorm(xadv.grad)
        delta = self.eps * grad
        # xadv = xadv + self.eps * grad
        
        if isinstance(self.clip_min, float):
            delta = clamp(xadv + delta, self.clip_min, self.clip_max) - xadv
        else:
            delta = channel_clip(xadv + delta, self.clip_min, self.clip_max) - xadv
            
        ### Update April 15
        ### Numerical instability problem when using Lp_norm pooling (for BPDA) 
        ### with large p (e.g. p >10), that results in overflow
        ### leading to misleading result as eps * NaN will effectively set xadv to be NaN
        ### if there was NaN in delta.data, then due to clipping operation
        ### the value will saturate to clip_min or clip_max, resulting in larger than epsilon distortion
        ### following line will cap the distortion size to epsilon again 
        ### however, note that this will result in a uniform noise that is not effective as adversarial attack
        ### SO IT IS IMPORTANT TO KEEP NUMERICAL STABILITY WHEN USING LPPOOL FOR BPDA
        ### (especially when using large complex dataset & model such as those for places365 or full-scale imagenet)
        ### (otherwise it will ruin the adversarial example, no effective!)
        delta.data = clamp_by_pnorm(delta.data, 2, self.eps)
        
        xadv = xadv + delta

        return xadv


FGM = GradientAttack

class RandomGradientAttack(Attack, LabelMixin):

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False):
        
        super(RandomGradientAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        
        if not isinstance(eps, float):
            eps = (eps[0] + eps[1] + eps[2]) / 3.

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None, temperature_softmax=False, temperature_softmax_value=None):
        x, y = self._verify_and_process_inputs(x, y)
        delta = torch.randn(x.size(), device=x.device)
        delta = normalize_by_pnorm(delta)
        delta = nn.Parameter(delta)
        delta.data = batch_multiply(self.eps/2, delta.data)
        
        if isinstance(self.clip_min, float):
            delta.data = clamp(x.data+delta.data, min=self.clip_min, max=self.clip_max) - x.data
        else:
            delta.data = channel_clip(x.data+delta.data, self.clip_min, self.clip_max) - x.data
        delta = delta.requires_grad_()

        #xadv = x.requires_grad_()
        outputs = self.predict(x + delta)
        if temperature_softmax:
            if temperature_softmax_value:
                outputs = outputs / temperature_softmax_value
            else:
                outputs = outputs / torch.std(outputs.clone().detach(), dim=1, keepdim=True)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        #grad_sign = delta.grad.detach().sign()
        grad = normalize_by_pnorm(delta.grad)
        delta.data = delta.data + batch_multiply(self.eps/2, grad)
        
        if isinstance(self.clip_min, float):
            delta.data = clamp(x.data+delta.data, self.clip_min, self.clip_max) - x.data
        else:
            delta.data = channel_clip(x.data+delta.data, self.clip_min, self.clip_max) - x.data
            
        delta.data = clamp_by_pnorm(delta.data, 2, self.eps)
        
        xadv = x + delta

        return xadv


RFGM = RandomGradientAttack