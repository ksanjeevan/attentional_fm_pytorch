import torch.nn.functional as F

def nll_loss(output, target):
    # loss for log_softmax
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    return F.cross_entropy(output, target)

def mse(output, target):
    return F.mse_loss(output, target)