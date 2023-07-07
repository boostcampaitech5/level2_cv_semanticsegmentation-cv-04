
# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# focal loss 
class Focal_loss(nn.Module):
    def __init__(self) -> None:
        super(Focal_loss, self).__init__()
        
    def forward(selt, inputs, targets, alpha=.25, gamma=2) : 
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        loss = alpha * (1-BCE_EXP)**gamma * BCE
        return loss 

class Dice_loss(nn.Module):
    def __init__(self) -> None:
        super(Dice_loss, self).__init__()
        
    def forward(self, pred, target, smooth = 1.):
        pred = pred.contiguous()
        target = target.contiguous()   
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
        return loss.mean()

class IOU_loss(nn.Module):
    def __init__(self) -> None:
        super(IOU_loss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1) : 
        inputs = F.sigmoid(inputs)      
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        IoU = (intersection + smooth)/(union + smooth)
        return 1 - IoU

class Calc_loss(nn.Module):
    def __init__(self) -> None:
        super(Calc_loss, self).__init__()
    
    def dice_loss(self, pred, target, smooth = 1.):
        pred = pred.contiguous()
        target = target.contiguous()   
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
        return loss.mean()
    
    def forward(self, pred, target, bce_weight = 0.5):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = F.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)
        return loss
