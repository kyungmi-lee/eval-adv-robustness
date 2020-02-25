import torch
import torch.nn as nn 
#from torch.autograd import Variable

def jacobian_regularization(net, data, target, loss_fn, p=2):
    data = data.requires_grad_()
    #target = target.clone().detach()
    output = net(data)
    classification_loss = loss_fn(output, target)
    grad = torch.autograd.grad(classification_loss, data, create_graph=True)[0]
    loss = torch.norm(grad.view(data.size(0), -1), p=p, dim=1)
    if p == 2:
        loss = loss.pow(2)
    return loss.sum()
