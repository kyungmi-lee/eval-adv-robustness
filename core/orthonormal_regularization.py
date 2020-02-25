import torch
import torch.nn as nn 
#from torch.autograd import Variable

def orthonormal_regularization(net, order=2, double=False):
    loss = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            w = layer.weight
            m = w.view(w.shape[0], -1)
            I = torch.eye(m.shape[0], device=w.device)
            d = (torch.matmul(m, m.t()) - I).norm(p=order)
            if order == 2 or order == 'fro':
                d = torch.pow(d, 2)
            loss += d

            if double:
                I_t = torch.eye(m.shape[1], device=w.device)
                d_t = (torch.matmul(m.t(), m) - I_t).norm(p=order)
                if order == 2 or order == 'fro':
                    d_t = torch.pow(d_t, 2)
                loss += d_t

    return loss