import torch.nn.functional as F
from . import functions as Func
import torch.nn as nn
import numpy as np
import torch


class FocalOnehotLoss(nn.Module):
    __name__ = 'FocalOnehotLoss'
    def __init__(self, alpha=2, reduction="mean"):
        super(FocalOnehotLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logit, onehot):
        batch_size = logit.shape[0]
        probability = F.softmax(logit,1)
        weight = (1-probability)**self.alpha
        weight = weight/(onehot*weight).sum().item()*batch_size

        log_probability = -F.log_softmax(logit,1)
        loss = (log_probability*onehot*weight)
        loss = loss.sum(1)
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            loss = loss
        else:
            raise NotImplementedError
        return loss


class CrossEntropyOnehotLoss(nn.Module):
    __name__ = 'CrossEntropyOnehotLoss'
    def __init__(self, reduction="mean"):
        super(CrossEntropyOnehotLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logit, onehot):
        log_probability = -F.log_softmax(logit,1)
        loss = (log_probability*onehot)
        loss = loss.sum(1)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            loss = loss
        else:
            raise NotImplementedError
        return loss
    
    
class CrossEntropyOnehotLossOHEM(nn.Module):
    def __init__(self, top_k=0.5):
        super(CrossEntropyOnehotLossOHEM, self).__init__()
        self.top_k = top_k
        self.loss = CrossEntropyOnehotLoss(reduction="none")

    def forward(self, input, target):
        loss = self.loss(input, target)
        if self.top_k == 1:
            return torch.mean(loss)
        else:
            valid_loss, idxs = torch.topk(loss, int(self.top_k * loss.size()[0]), dim=0)    
            return torch.mean(valid_loss)
        
class FocalOnehotLossOHEM(nn.Module):
    def __init__(self, top_k=0.5):
        super(FocalOnehotLossOHEM, self).__init__()
        self.top_k = top_k
        self.loss = FocalOnehotLoss(reduction="none")

    def forward(self, input, target):
        loss = self.loss(input, target)
        if self.top_k == 1:
            return torch.mean(loss)
        else:
            valid_loss, idxs = torch.topk(loss, int(self.top_k * loss.size()[0]), dim=0)    
            return torch.mean(valid_loss)


class SoftDiceLoss_binary(nn.Module):
    def __init__(self):
        super(SoftDiceLoss_binary, self).__init__()

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input = F.sigmoid(input).view(batch_size, -1)
        # print(target.shape)
        # print(target.view(-1))
        target = target.clone().view(batch_size, -1)

        inter = torch.sum(input * target, 1) + smooth
        union = torch.sum(input * input, 1) + torch.sum(target * target, 1) + smooth

        score = torch.sum(2.0 * inter / union) / float(batch_size)
        score = 1.0 - torch.clamp(score, 0.0, 1.0 - 1e-7)

        return score


class WeightedBCELoss(nn.Module):
    __name__ = 'WeightedBCELoss'
    def __init__(self, weight):
        super(WeightedBCELoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        c = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.weight))
        return c(input, target)
    
    
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def hinge(pred, label):
    signs = 2 * label - 1
    errors = 1 - pred * signs
    return errors


def lovasz_hinge_flat(logits, labels, ignore_index):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    logits = logits.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    
    errors = hinge(logits, labels)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss


class LovaszLoss(nn.Module):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    __name__ = 'LovaszLoss'
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        return lovasz_hinge_flat(logits, labels, self.ignore_index)
    

# class FocalLoss(nn.Module):
#     __name__ = 'FocalLoss'
#     def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=None):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.weight = weight
#         self.ignore_index = ignore_index
#         self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

#     def forward(self, preds, labels):
#         if self.ignore_index is not None:
#             mask = labels != self.ignore_index
#             labels = labels[mask]
#             preds = preds[mask]

#         logpt = -self.bce_fn(preds, labels)
#         pt = torch.exp(logpt)
#         loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
#         return loss
    
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()



class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - Func.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - Func.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)


class BCEJaccardLoss(JaccardLoss):
    __name__ = 'bce_jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        jaccard = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce




