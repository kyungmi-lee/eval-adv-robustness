# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch

import sys
sys.path.insert(0, './advertorch')

from advertorch.utils import batch_multiply
from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin

# Baseline attacks implemented at advertorch
from advertorch.attacks import FGSM as FGSM_base
from advertorch.attacks import FGM  as FGM_base

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
        delta.data = batch_multiply(self.eps/2, delta.data)
        delta.data = clamp(x.data+delta.data, min=self.clip_min, max=self.clip_max) - x
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
        delta.data = delta.data + batch_multiply(self.eps/2, grad_sign)
        delta.data = clamp(x.data+delta.data, self.clip_min, self.clip_max) - x.data

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

        xadv = xadv + self.eps * grad_sign

        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv


FGSM = GradientSignAttack

class GradientAttack(FGM_base):
    
    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0., clip_max=1., targeted=False):
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
        xadv = xadv + self.eps * grad
        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv


FGM = GradientAttack

class RandomGradientAttack(Attack, LabelMixin):

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False):
        
        super(RandomGradientAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

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
        delta.data = clamp(x.data+delta.data, min=self.clip_min, max=self.clip_max) - x
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
        delta.data = clamp(x.data+delta.data, self.clip_min, self.clip_max) - x.data
        xadv = x + delta

        return xadv


RFGM = RandomGradientAttack