# Utility functions for channel-dependent eps and clip_min, clip_max 
# associated with normalizing each channel with different mean, stdev

import torch

def channel_eps_multiply(grad_sign, eps, scale=None):
    if not isinstance(eps, torch.Tensor):
        eps_tensor = torch.Tensor(eps)
    else:
        eps_tensor = eps
    eps_tensor = eps_tensor.to(grad_sign.device)
    if scale:
        eps_tensor = eps_tensor * scale
    eps_tensor = eps_tensor.view((1, 3, 1, 1))
    eps_tensor = eps_tensor.repeat(grad_sign.shape[0], 1, grad_sign.shape[2], grad_sign.shape[3])
    
    return torch.mul(eps_tensor, grad_sign)

def channel_clip(xadv, clip_min, clip_max):
    l = torch.Tensor(clip_min).to(xadv.device)
    l = l.view((1, 3, 1, 1))
    l = l.repeat(xadv.shape[0], 1, xadv.shape[2], xadv.shape[3])
    
    u = torch.Tensor(clip_max).to(xadv.device)
    u = u.view((1, 3, 1, 1))
    u = u.repeat(xadv.shape[0], 1, xadv.shape[2], xadv.shape[3])
    
    return torch.max(torch.min(xadv, u), l)

def channel_eps_clip(x, eps):
    if not isinstance(eps, torch.Tensor):
        eps_tensor = torch.Tensor(eps).to(x.device)
    else:
        eps_tensor = eps
    eps_tensor = eps_tensor.to(x.device)
    eps_tensor = eps_tensor.view((1, 3, 1, 1))
    eps_tensor = eps_tensor.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
    
    return torch.max(torch.min(x, eps_tensor), -eps_tensor) 