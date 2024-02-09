import torch

def compute_iou(targets, predictions):

    targets = targets.view(-1)
    predictions = predictions.view(-1)

    intersection = torch.sum(predictions == targets) # true positives
    union = len(predictions) + len(targets) - intersection

    return intersection / union 