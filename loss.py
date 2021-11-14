"""
@authors: Helin Wang, Dongchao Yang
@Introduction: Loss
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLossV1(nn.Module):
 
    def __init__(self,
                alpha=0.25,
                gamma=2,
                reduction='mean',):
        
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # self.crit = nn.BCEWithLogitsLoss(reduction='none')

        # self.celoss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, label):
        # print('....')
        # print(logits.shape)
        '''
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha
        ce_loss=(-(label * torch.log(logits)) - (
                    (1 - label) * torch.log(1 - logits)))
        # print(ce_loss)
        # ce_loss=(-(label * torch.log(torch.softmax(logits, dim=1))) - (
        #             (1 - label) * torch.log(1 - torch.softmax(logits, dim=1))))
        pt = torch.where(label == 1, logits, 1 - logits)
        # print('pt ',pt)
        # ce_loss = self.crit(logits, label)
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        # print('loss ',loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def get_loss(output, target):
    '''BCE loss

    Args:
      output: (N)
      target: (N)
    '''

    loss_function = torch.nn.BCELoss()
    loss = loss_function(output, target.squeeze())
    return loss

def focal_loss(output,target):
    loss_function = FocalLossV1()
    loss = loss_function(output,target)
    return loss
