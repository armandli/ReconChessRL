import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class WeightedBCELoss(nn.Module):
  def __init__(self, reduction='mean'):
    super(WeightedBCELoss, self).__init__()
    self.reduction = reduction
  def forward(self, pred, target, weight):
    return F.binary_cross_entropy(pred, target, weight=weight, reduction=self.reduction)

class SmoothedBCELoss(nn.Module):
  def __init__(self, epsilon, reduction='mean'):
    super(SmoothedBCELoss, self).__init__()
    self.epsilon = epsilon
    self.reduction = reduction
  def forward(self, pred, target):
    with torch.no_grad():
      target = target * (1. - 2. * self.epsilon) + self.epsilon
    return F.binary_cross_entropy(pred, target, reduction=self.reduction)

class WeightedSmoothedBCELoss(nn.Module):
  def __init__(self, epsilon, reduction='mean'):
    super(WeightedSmoothedBCELoss, self).__init__()
    self.epsilon = epsilon
    self.reduction = reduction
  def forward(self, pred, target, weight):
    with torch.no_grad():
      target = target * (1. - 2. * self.epsilon) + self.epsilon
    return F.binary_cross_entropy(pred, target, weight=weight, reduction=self.reduction)

class WeightedMSELoss(nn.Module):
  def __init__(self, reduction='mean'):
    super(WeightedMSELoss, self).__init__()
    self.reduction = reduction

  def forward(self, pred, target, weight):
    loss = F.mse_loss(pred, target, reduction='none') * weight
    if self.reduction == 'mean':
      return torch.mean(loss)
    elif self.reduction == 'sum':
      return torch.sum(loss)
    else:
      return loss

class TDMSEError(nn.Module):
  def __init__(self):
    super(TDMSEError, self).__init__()

  def forward(self, pred, target, weight, epsilon):
    loss = torch.mean(F.mse_loss(pred, target, reduction='none') * weight)
    err = F.l1_loss(pred, target, reduction='none') + epsilon
    return (loss, err)

class PGError(nn.Module):
  def __init__(self):
    super(PGError, self).__init__()

  def forward(self, pi, r, epsilon):
    with torch.no_grad():
      r_mean = torch.mean(r)
    l = torch.mean(-1. * torch.log(pi + epsilon) * (r - r_mean))
    return l

class QRHuberError(nn.Module):
  def __init__(self, quantiles):
    super(QRHuberError, self).__init__()
    self.tau = torch.tensor((2. * np.arange(quantiles) + 1) / (2. * quantiles), dtype=torch.float32).view(1, -1)

  def forward(self, pred, target, kappa):
    with torch.no_grad():
      quantile_loss = torch.abs(self.tau - ((target - pred) < 0.).to(dtype=torch.float32))
    #NOTE: huber_loss is not implemented in pytorch yet
    l = torch.mean(torch.where(torch.abs(target - pred) < kappa, 0.5 * torch.pow(target - pred, 2.), kappa * (torch.abs(target - pred) - 0.5 * kappa)) * quantile_loss)
    return l

class QRWeightedHuberError(nn.Module):
  def __init__(self, quantiles):
    super(QRWeightedHuberError, self).__init__()
    self.tau = torch.tensor((2. * np.arange(quantiles) + 1) / (2. * quantiles), dtype=torch.float32).view(1, -1)

  def forward(self, pred, target, weight, kappa, epsilon):
    with torch.no_grad():
      quantile_loss = torch.abs(self.tau - ((target - pred) < 0.).to(dtype=torch.float32))
    #NOTE: huber_loss is not implemented in pytorch yet
    # dimension of loss (b, 1, Q)
    loss = torch.where(torch.abs(target - pred) < kappa, 0.5 * torch.pow(target - pred, 2.), kappa * (torch.abs(target - pred) - 0.5 * kappa))
    l = torch.mean(loss * quantile_loss * weight)
    #NOTE: I think we can use either sum or mean here
    err = (torch.sum(loss, dim=2, keepdim=True) + epsilon).reshape((-1))
    return (l, err)

class IQNHuberError(nn.Module):
  def __init__(self):
    super(IQNHuberError, self).__init__()

  def forward(self, pred, target, tau, kappa):
    with torch.no_grad():
      quantile_loss = torch.abs(tau - ((target - pred) < 0.).to(dtype=torch.float32))
    #NOTE: huber_loss is not implemented in pytorch yet
    l = torch.mean(torch.where(torch.abs(target - pred) < kappa, 0.5 * torch.pow(target - pred, 2.), kappa * (torch.abs(target - pred) - 0.5 * kappa)) * quantile_loss)
    return l

class IQNWeightedHuberError(nn.Module):
  def __init__(self):
    super(IQNWeightedHuberError, self).__init__()

  def forward(self, pred, target, tau, weight, kappa, epsilon):
    with torch.no_grad():
      quantile_loss = torch.abs(tau - ((target - pred) < 0.).to(dtype=torch.float32))
    #NOTE: huber_loss is not implemented in pytorch yet
    #dimension of loss (b, T, T)
    loss = torch.where(torch.abs(target - pred) < kappa, 0.5 * torch.pow(target - pred, 2.), kappa * (torch.abs(target - pred) - 0.5 * kappa))
    l = torch.mean(loss * quantile_loss * weight)
    #NOTE: I think we can use either sum or mean here
    err = torch.sum(torch.sum(dim=2), dim=1) + epsilon
    return (l, err)