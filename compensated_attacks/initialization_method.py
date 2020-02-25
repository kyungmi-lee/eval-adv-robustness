import torch
import torch.nn as nn
import numpy as np

def bfgs_direction(xvar, yvar, predict, eps_iter, loss_fn, clip_min=0.0, clip_max=1.0, targeted=False, norm=2):
   
    _xvar = xvar.clone().detach().requires_grad_()
    
    outputs = predict(_xvar)
    loss = loss_fn(outputs, yvar)
    if targeted:
        loss = -loss
    loss.backward()
    grad = _xvar.grad.detach()
    
    identity = torch.eye(xvar.size(1) * xvar.size(2) * xvar.size(3), device=xvar.device).unsqueeze_(0).repeat(xvar.size(0), 1, 1)
    delta = grad
    delta = torch.div(delta, torch.norm(delta.view(xvar.size(0), -1), p=2, dim=1).view(xvar.size(0), 1, 1, 1).expand_as(grad)+1e-8)
    delta = delta * eps_iter
    delta = torch.clamp(xvar+delta.view_as(xvar), min=clip_min, max=clip_max) - xvar
    
    new_xvar = (xvar.clone().detach()+delta.clone().detach()).requires_grad_()
    new_outputs = predict(new_xvar)
    loss = loss_fn(new_outputs, yvar)
    if targeted:
        loss = -loss
    loss.backward()
    new_grad = new_xvar.grad.detach()
    
    y = (new_grad - grad).view(xvar.size(0), xvar.size(1)*xvar.size(2)*xvar.size(3), 1)
    delta = delta.view_as(y)
    
    dy = torch.bmm(torch.transpose(delta, 1, 2), y).squeeze()
    _a = torch.bmm(y, torch.transpose(delta, 1, 2))
    _a = torch.div(_a, dy.view(xvar.size(0), 1, 1).expand_as(_a))
    a = identity - _a
    b = torch.bmm(delta, torch.transpose(delta, 1, 2))
    b = torch.div(b, dy.view(xvar.size(0), 1, 1).expand_as(b))
    #print(dy.shape, a.shape, b.shape)
    
    hess_inv = torch.bmm(torch.bmm(torch.transpose(a, 1, 2), identity), a) + b
    #print(hess_inv.shape)
    
    new_delta = torch.bmm(hess_inv, grad.view(xvar.size(0), -1, 1)).squeeze(dim=2)
    #print(new_delta.shape, mask.shape)
    new_delta = torch.div(new_delta, torch.norm(new_delta.view(xvar.size(0), -1), p=2, dim=1).\
                          view(xvar.size(0),1).expand_as(new_delta)+1e-8)
    if norm == np.inf:
        eps_equiv = np.sqrt(xvar.size(1)*xvar.size(2)*xvar.size(3)/3.14) * eps_iter
        new_delta = new_delta * eps_equiv
        new_delta = torch.clamp(new_delta, min=-eps_iter, max=eps_iter)
    else:
        new_delta = new_delta * eps_iter
    new_delta = new_delta.view_as(xvar).data
    #new_delta = (1.0 - mask) * new_delta + mask * (-new_delta)
    
    return new_delta

def miyato_second_order(xvar, yvar, predict, eps, loss_fn, delta=0.5, clip_min=0.0, clip_max=1.0, targeted=False, norm=2):
    rand_vector = torch.randn_like(xvar)
    rand_vector /= torch.norm(rand_vector.view(xvar.size(0), -1), dim=1, p=2).view(xvar.size(0), 1, 1, 1).expand_as(xvar)
    rand_vector *= delta
    
    x = xvar.clone().detach().requires_grad_()
    output = predict(x)
    loss = loss_fn(output, yvar)
    if targeted:
        loss = -loss
    loss.backward()
    grad = x.grad.detach().data
    
    rand_vector = torch.clamp(rand_vector+xvar, min=clip_min, max=clip_max) - rand_vector
    rand_vector = nn.Parameter(rand_vector)
    rand_vector = rand_vector.requires_grad_()
    output = predict(xvar.clone()+rand_vector)
    loss = loss_fn(output, yvar)
    if targeted:
        loss = -loss
    loss.backward()
    grad_rand = rand_vector.grad.detach().data
    
    hd = grad_rand - grad
    hd /= torch.norm(hd.view(xvar.size(0), -1), dim=1, p=2).view(xvar.size(0), 1, 1, 1).expand_as(grad)
    if norm == np.inf:
        # https://arxiv.org/pdf/1805.12514.pdf 
        # e_2 = sqrt(d/pi) * e_inf
        eps_equiv = np.sqrt(xvar.size(1)*xvar.size(2)*xvar.size(3)/3.14) * eps
        init = eps_equiv * hd
        # Trim by limiting to the linf-norm box
        init = torch.clamp(init, min=-eps, max=eps)
        return init
    else:
        return eps * hd

