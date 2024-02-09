import warnings
warnings.simplefilter('ignore')

import torch
from torch.utils.data import DataLoader

import numpy as np

from dataset.read_stl_part import Obj_dataset_adj, collect_npy
from model.pointnet import Pointnet_glm
from model.loss_vis import drow
from model.pytorchtools import EarlyStopping
from model.loss_fn import FocalDiceDsLoss

from utils.tool import folder_check
from utils.iou import compute_iou

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

n_epochs = 100
n_point = 4096
num_cls = 15
batch_size = 24

train_data_path = "part_teeth/train_low_npy/"
valid_data_path = "part_teeth/valid_low_npy/"

pointnet_glm = Pointnet_glm(n_point, n_cls=num_cls).to(device)

optimizer = torch.optim.Adam(pointnet_glm.parameters(), lr=1e-4, weight_decay=1e-3, eps=1e-3)

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, 
                                              step_size_up=1000, cycle_momentum=False)

alpha = np.ones(num_cls)
alpha[0] *= 0.25 # balance background classes
gamma = 1

criterion = FocalDiceDsLoss(alpha=alpha, gamma=gamma, dice=True).to(device)

early_stopping = EarlyStopping( patience=20,
                                verbose=False,
                                save_model=False)

def torch_data():

    all_train = collect_npy(train_data_path)
    all_vaild = collect_npy(valid_data_path)

    train_data = Obj_dataset_adj(n_point, all_train, num_cls, data_path=train_data_path, isAug=False)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True,
        pin_memory=True
        )

    valid_data = Obj_dataset_adj(n_point, all_vaild, num_cls, data_path=valid_data_path, isAug=False)
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=True
        )
    
    return train_loader, valid_loader

def train(model, vert, mask, adj, y_ohe):

    optimizer.zero_grad()
    preds = model(vert, adj)
    
    pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)

    loss = criterion(preds, mask, y_ohe, pred_choice)
    
    loss.backward()
    
    optimizer.step()
    scheduler.step()
    
    correct = pred_choice.eq(mask.data).cpu().sum()
    accuracy = correct/float(batch_size*n_point)
    iou = compute_iou( mask, pred_choice)

    return loss, accuracy, iou

def valid(model, vert, mask, adj, y_ohe):
    preds = model(vert, adj)
    pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
    
    loss = criterion(preds, mask, y_ohe, pred_choice)
    
    correct = pred_choice.eq(mask.data).cpu().sum()
    accuracy = correct/float(batch_size*n_point)
    iou = compute_iou( mask, pred_choice)

    return loss, accuracy, iou
    
if __name__ == "__main__":

    folder_check()

    train_loader, valid_loader = torch_data()
    
    i_iter = 0
    best_iou = 0

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    num_batches = len(train_loader)

    # pointnet_glm = torch.compile(pointnet_glm)

    for epoch in range(n_epochs):

        b_train_loss = []
        b_valid_loss = []
        b_train_acc = []
        b_valid_acc = []
        b_train_iou = []
        b_valid_iou = []

        pointnet_glm.train()

        for idx, (vert, mask, y_ohe) in enumerate(train_loader):
            vert = vert.to(device)
            mask = mask.to(device)
            y_ohe = y_ohe.to(device)

            
            adj = torch.cdist(vert, vert)
            adj = torch.where(adj < 0.1, 1., adj)

            loss, accuracy, iou = train(pointnet_glm, vert, mask, adj, y_ohe)
            
            b_train_loss.append(loss.item())
            b_train_acc.append(accuracy)
            b_train_iou.append(iou.item())

        ep_train_loss = np.mean(b_train_loss)
        ep_train_acc = np.mean(b_train_acc)
        ep_train_iou = np.mean(b_train_iou)

        ################ validation #########################
        pointnet_glm.eval()
        with torch.no_grad():
            for _, (vert, mask, y_ohe) in enumerate(valid_loader):
                mask = mask.to(device)
                vert = vert.to(device)
                y_ohe = y_ohe.to(device) # for FocalDiceDsLoss

                adj = torch.cdist(vert, vert)
                adj = torch.where(adj < 0.1, 1., adj)

                loss, accuracy, iou = valid(pointnet_glm, vert, mask, adj, y_ohe)

                b_valid_loss.append(loss.item())
                b_valid_acc.append(accuracy)
                b_valid_iou.append(iou.item())

        ################ loogging #########################
        ep_valid_loss = np.mean(b_valid_loss)
        ep_valid_acc = np.mean(b_valid_acc)
        ep_valid_iou = np.mean(b_valid_iou)


        train_losses.append(ep_train_loss)
        valid_losses.append(ep_valid_loss)
        train_accs.append(ep_train_acc)
        valid_accs.append(ep_valid_acc)

        print(f'{epoch + 1:2d}/{n_epochs:2d} {idx + 1:3d}/{len(train_loader):3d}, \
            train loss: {ep_train_loss:8.5f}, \
            train acc: {ep_train_acc:7.5f}, \
            train iou: {ep_train_iou:7.5f}, \
            valid loss: {ep_valid_loss:8.5f}, \
            valid acc: {ep_valid_acc:7.5f},\
            valid iou: {ep_valid_iou:7.5f}'
            )

        
        if ep_valid_iou >= best_iou:
            best_iou = ep_valid_iou
            torch.save(pointnet_glm.state_dict(), "saved_model/pointnet_glm.pth")

        early_stopping(ep_valid_loss, pointnet_glm)

        if early_stopping.early_stop:
            print("[INFO] Early stopping")
            break
        i_iter += 1

    drow(train_losses, valid_losses, train_accs, valid_accs, "part_lower_glm")