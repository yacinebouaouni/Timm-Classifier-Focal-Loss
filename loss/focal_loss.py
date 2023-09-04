import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.weight = None
        self.avg_factor = None

   
    def forward(self, pred, target):

        """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

        Different from `py_sigmoid_focal_loss`, this function accepts probability
        as input.
        Args:
            pred (torch.Tensor): The prediction probability with shape (N, C),
                C is the number of classes.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """

        target = target.type_as(pred)

        pt = pred * target + (1-pred) * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *(1 - target)) * (1.0-pt).pow(self.gamma)

        loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
        if self.weight is not None:
            if self.weight.shape != loss.shape:
                if self.weight.size(0) == loss.size(0):
                    # For most cases, weight is of shape (num_priors, ),
                    #  which means it does not have the second axis num_class
                    self.weight = self.weight.view(-1, 1)
                else:
                    # Sometimes, weight per anchor per class is also needed. e.g.
                    #  in FSAF. But it may be flattened of shape
                    #  (num_priors x num_class, ), while loss is still of shape
                    #  (num_priors, num_class).
                    assert self.weight.numel() == loss.numel()
                    self.weight = self.weight.view(loss.size(0), -1)
            assert self.weight.ndim == loss.ndim
        loss = self.weight_reduce_loss(loss, self.weight, self.reduction, self.avg_factor)
        return loss





    def weight_reduce_loss(self, loss, weight=None, reduction='mean', avg_factor=None):
        """Apply element-wise weight and reduce loss.
        Args:
            loss (Tensor): Element-wise loss.
            weight (Tensor): Element-wise weights.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Average factor when computing the mean of losses.
        Returns:
            Tensor: Processed loss values.
        """
        # if weight is specified, apply element-wise weight
        if weight is not None:
            loss = loss * weight

        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            loss = self.reduce_loss(loss, reduction)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                # Avoid causing ZeroDivisionError when avg_factor is 0.0,
                # i.e., all labels of an image belong to ignore index.
                eps = torch.finfo(torch.float32).eps
                loss = loss.sum() / (avg_factor + eps)
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss
    
        
    def reduce_loss(self, loss, reduction):
        """Reduce loss as specified.
        Args:
            loss (Tensor): Elementwise loss tensor.
            reduction (str): Options are "none", "mean" and "sum".
        Return:
            Tensor: Reduced loss tensor.
        """
        reduction_enum = F._Reduction.get_enum(reduction)
        # none: 0, elementwise_mean:1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()
            