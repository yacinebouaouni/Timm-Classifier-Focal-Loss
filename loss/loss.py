import torch.nn as nn
from . import focal_loss


def select_loss(config):
    loss_type = config.get("loss_type")

    if loss_type == "focal":
        alpha = config.get("alpha", 0.5)
        gamma = config.get("gamma", 2)
        reduction = config.get("reduction", "sum")
        return focal_loss.FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
    elif loss_type == "cross_entropy":
        reduction = config.get("reduction", "sum")
        return nn.CrossEntropyLoss(reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")