import os
import logging
import numpy as np

import torch
from torch.nn import init

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
            
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

import torch

def metrics(output, target, topk=(1,)):
    """
    Computes MCC, F1-score, Recall, and Precision for the specified values of topk.
    
    Args:
        output (torch.Tensor): Model outputs (logits or probabilities) of shape (batch_size, num_classes).
        target (torch.Tensor): Ground truth labels of shape (batch_size).
        topk (tuple): Tuple of top-k values to evaluate.
        
    Returns:
        dict: A dictionary containing MCC, F1-score, Recall, and Precision for each k in topk.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # Get top-k predictions
    pred = pred.t()  # Transpose to shape (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # Compare with ground truth

    metrics_res = {k: {'MCC': 0, 'F1': 0, 'Recall': 0, 'Precision': 0} for k in topk}

    for k in topk:
        # Calculate True Positives, False Positives, False Negatives, and True Negatives
        pred_k = pred[:k]  # Top-k predictions
        pred_flat = pred_k.reshape(-1)  # Flatten predictions
        target_expand = target.view(1, -1).expand_as(pred_k).reshape(-1)  # Flatten ground truth

        tp = (pred_flat == target_expand).sum().item()  # True Positives
        fp = (pred_flat != target_expand).sum().item()  # False Positives
        fn = batch_size * k - tp  # False Negatives
        tn = batch_size * maxk - tp - fp - fn  # True Negatives

        # Avoid division by zero with eps
        eps = 1e-7

        # Matthews Correlation Coefficient
        numerator = (tp * tn - fp * fn)
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = numerator / (denominator + eps)

        # Precision, Recall, F1-score
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)

        # Save metrics for the current k
        metrics_res[k] = {
            'MCC': mcc,
            'F1': f1_score * 100,
            'Recall': recall * 100,
            'Precision': precision * 100
        }

    return metrics_res


def norm(x):

    n = np.linalg.norm(x)
    return x / n
