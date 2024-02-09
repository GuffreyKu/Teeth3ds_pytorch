import torch
import torch.nn.functional as F
import numpy as np

class FocalDiceLoss(torch.nn.Module): #Focal Loss + Dice Loss
    def __init__(self, alpha=None, gamma=0, size_average=True, dice=False):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.dice = dice

        # sanitize inputs
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,(list, np.ndarray)): self.alpha = torch.Tensor(alpha)

        # get Balanced Cross Entropy Loss
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=self.alpha)
        
    def forward(self, predictions, targets, pred_choice=None):

        # get Balanced Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions.transpose(2, 1), targets)
        
        predictions = predictions.contiguous() \
                                 .view(-1, predictions.size(2))
         
        # get predicted class probabilities for the true class
        pn = F.softmax(predictions)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # compute loss (negative sign is included in ce_loss)
        loss = ((1 - pn)**self.gamma * ce_loss)
        if self.size_average: loss = loss.mean() 
        else: loss = loss.sum()

        if self.dice: return loss + self.dice_loss(targets, pred_choice, eps=1)
        else: return loss

    @staticmethod
    def dice_loss(predictions, targets, eps=1):
        ''' Compute Dice loss, directly compare predictions with truth '''

        targets = targets.reshape(-1)
        predictions = predictions.reshape(-1)

        cats = torch.unique(targets)

        top = 0
        bot = 0
        for c in cats:
            locs = targets == c

            # get truth and predictions for each class
            y_tru = targets[locs]
            y_hat = predictions[locs]

            top += torch.sum(y_hat == y_tru)
            bot += len(y_tru) + len(y_hat)

        return 1 - 2*((top + eps)/(bot + eps))
    
class CeRegloss(torch.nn.Module):
    def __init__(self, weight, device, num_cls):
        super(CeRegloss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.device = device
        self.num_cls = num_cls

    def feature_transform_reguliarzer(self,trans):
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        I = I.to(self.device)
        loss = F.mse_loss(torch.bmm(trans, trans.transpose(2, 1)), I)
        return loss
    
    def forward(self, pred, target, trans_feat):

        target = target.view(-1)
        pred = pred.view(-1, self.num_cls)

        loss = self.loss(pred, target)
        mat_diff_loss = self.feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * 1e-3

        return total_loss
    

class FocalDiceDsLoss(torch.nn.Module): #Focal Loss + Dice Loss + Disence KL Loss
    def __init__(self, alpha=None, gamma=0, size_average=True, dice=False):
        super(FocalDiceDsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.dice = dice

        # sanitize inputs
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,(list, np.ndarray)): self.alpha = torch.Tensor(alpha)

        # get Balanced Cross Entropy Loss
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=self.alpha)
        

    def forward(self, predictions, targets, y_ohe, pred_choice=None):
        # get Balanced Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions.transpose(2, 1), targets)
        # get distance kl divergence
        dis_map = torch.cdist(y_ohe, y_ohe)
        dis_map = torch.where(dis_map > 1, 0.9, dis_map)
        dis_map = torch.where(dis_map < 1, 0.05, dis_map)

        pred_prob = F.softmax(predictions, dim=-1)
        pred_map = torch.cdist(pred_prob, pred_prob)
        
        dis_map_loss = F.kl_div(pred_map, dis_map)

        # reformat predictions (b, n, c) -> (b*n, c)
        predictions = predictions.contiguous() \
                                 .view(-1, predictions.size(2))
         
        # get predicted class probabilities for the true class
        pn = F.softmax(predictions)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # compute loss (negative sign is included in ce_loss)
        loss = ((1 - pn)**self.gamma * ce_loss)
        if self.size_average: loss = loss.mean() 
        else: loss = loss.sum()

        # add dice coefficient if necessary
        if self.dice: return loss + self.dice_loss(targets, pred_choice, eps=1) + dis_map_loss
        else: return loss
    
    @staticmethod
    def dice_loss(predictions, targets, eps=1):
        ''' Compute Dice loss, directly compare predictions with truth '''

        targets = targets.reshape(-1)
        predictions = predictions.reshape(-1)

        cats = torch.unique(targets)

        top = 0
        bot = 0
        for c in cats:
            locs = targets == c

            # get truth and predictions for each class
            y_tru = targets[locs]
            y_hat = predictions[locs]

            top += torch.sum(y_hat == y_tru)
            bot += len(y_tru) + len(y_hat)

        return 1 - 2*((top + eps)/(bot + eps))