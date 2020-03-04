from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, './advertorch')

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import rand_init_delta

from initialization_method import *

from advertorch.attacks import PGDAttack as PGD_base

# perturb_iterative modified for logit rescaling option; otherwise same as the original function in AdverTorch
# PGD modified for different *initialization* methods (PGD + Eigen and PGD + BFGS)

def perturb_iterative(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0, temperature_softmax=False, temperature=None):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size per iteration.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: (optional float) mininum value per input dimension.
    :param clip_max: (optional float) maximum value per input dimension.
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    for ii in range(nb_iter):
        outputs = predict(xvar + delta)
        if temperature_softmax:
            if temperature:
                outputs = outputs / temperature
            else:
                outputs = outputs / torch.std(outputs.clone().detach(), dim=1, keepdim=True)
        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data

        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)
        else:
            error = "Only ord = inf and ord = 2 have been implemented"
            raise NotImplementedError(error)

        delta.grad.data.zero_()

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    return x_adv



class PGDAttack(PGD_base):

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, targeted=False, init_method='rand'):
        """
        Create an instance of the PGDAttack.

        :param predict: forward pass function.
        :param loss_fn: loss function.
        :param eps: maximum distortion.
        :param nb_iter: number of iterations
        :param eps_iter: attack step size.
        :param rand_init: (optional bool) random initialization.
        :param clip_min: mininum value per input dimension.
        :param clip_max: maximum value per input dimension.
        :param ord: norm type of the norm constraints
        :param targeted: if the attack is targeted
        """
        super(PGDAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init, clip_min, clip_max, \
            ord, targeted)
        
        self.init_method = init_method
        
    def perturb(self, x, y=None, temperature_softmax=False, temperature=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        if self.rand_init:
            if self.init_method == 'rand':
                delta = nn.Parameter(delta)
                rand_init_delta(
                    delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            elif self.init_method == 'miyato':
                delta = miyato_second_order(x.clone(), y.clone(), self.predict, self.eps, self.loss_fn, delta=self.eps, \
                                            clip_min=self.clip_min, clip_max=self.clip_max, targeted=self.targeted, norm=self.ord)
                delta = nn.Parameter(delta)
            elif self.init_method == 'bfgs':
                delta = bfgs_direction(x.clone(), y.clone(), self.predict, self.eps, self.loss_fn, \
                                       clip_min=self.clip_min, clip_max=self.clip_max, targeted=self.targeted, norm=self.ord)
                delta = nn.Parameter(delta)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x
        else:
            delta = nn.Parameter(delta)

        rval = perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,\
            temperature_softmax=temperature_softmax, temperature=temperature)

        return rval.data


class LinfPGDAttack(PGDAttack):
    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, init_method='rand'):
        ord = np.inf
        super(LinfPGDAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init,
            clip_min, clip_max, ord, targeted, init_method)


class L2PGDAttack(PGDAttack):
    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, init_method='rand'):
        ord = 2
        super(L2PGDAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init,
            clip_min, clip_max, ord, targeted, init_method)
