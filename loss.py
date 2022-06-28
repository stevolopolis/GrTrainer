import torch
import torch.nn.functional as F


def CLSLoss(output, target):
    """l1_loss = F.l1_loss(output, target)
    output_sum = torch.sum(output, dim=1)
    sum_diff = torch.abs(1 - output_sum)
    sum_cond = sum_diff < 1
    sum_loss = - torch.log(torch.where(sum_cond, sum_diff, torch.ones_like(sum_diff, dtype=torch.float32)))
    loss = l1_loss + sum_loss"""
    loss = nll_loss(output, target)

    return torch.sum(loss) / loss.size(0)


def WeightedL2Loss(output, target):
    mse = torch.pow(output - target, 2)
    cond = [1, 1, 0, 0, 0]
    loss = torch.where(cond, 2*mse, mse)

    return torch.sum(loss) / loss.size(0)


def BCEL1Loss(output, target):
    """Returns BCELoss for when output is in [0, 1], and
    returns L1Loss for when output is not in [0, 1].

    Implemented using the format of smooth_l1_loss.
    """
    smaller_than_one = output < .9
    greater_than_zero = output > .1

    condition = smaller_than_one == greater_than_zero
    loss = torch.where(condition, nll_loss(output, target), torch.abs(output - target) + .1)
    
    return torch.sum(loss) / loss.size(0)


def nll_loss(output, target):
    return - (torch.log(output + 1e-5) * target + 0.1 * torch.log(1 - output + 1e-5) * (1 - target))