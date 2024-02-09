import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import os

from .aug import shift, jitter, rotate_point_cloud_x, rotate_point_cloud_y, rotate_point_cloud_z
import sys
sys.path.insert(0, '..')

from utils.sampling import sample_and_group



def collect_data(path):
    csv_datas = pd.read_csv(path)["path"].to_list()
    random.shuffle(csv_datas)
    return csv_datas

def collect_npy(path):
    npy_datas = os.listdir(path)
    if '.DS_Store' in npy_datas:
        npy_datas.remove('.DS_Store')

    random.shuffle(npy_datas)
    return npy_datas

class Obj_dataset(Dataset):
    def __init__(self, n_point, data, num_cls, data_path, isAug):
        self.datas = data
  
        self.n_point = n_point
        self.num_cls = num_cls
        self.isAug = isAug
        self.data_path = data_path
        self.max_cls = 15
        self.x = []
        self.y = []
        self.ohe = []

        for i in range(len(self.datas)):
            with open(self.data_path+self.datas[i], 'rb') as f:
                pc_data = np.load(f)
                label = np.load(f)
            f.close()

            if pc_data.shape[0] != self.n_point:
                continue

            pc_data = self.normalize_pc_min_max(pc_data)
            
            if self.isAug:
                pc_data = rotate_point_cloud_x(pc_data)
                pc_data = rotate_point_cloud_y(pc_data)
                pc_data = rotate_point_cloud_z(pc_data)
                pc_data = jitter(pc_data)
                pc_data = shift(pc_data, 0.1)
            
            y_ohe = np.zeros((len(label), self.max_cls+1))
            y_ohe[np.arange(len(label)), np.array(label)] = 1

            self.x.append(pc_data)
            self.y.append(label)
            self.ohe.append(y_ohe)

    def normalize_pc(self,points):
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance
        return points
    
    def normalize_pc_min_max(self,points):
        points -= points.min(axis=0)
        points /= points.max(axis=0)
        return points
    
    def __getitem__(self, index):

        pc_data = self.x[index]
        label = self.y[index]
        pc_data = torch.from_numpy(pc_data)
        label = torch.from_numpy(label)

        return pc_data, label

    def __len__(self):
        return len(self.x)
    

class Obj_dataset_adj(Dataset):
    def __init__(self, n_point, data, num_cls, data_path, isAug):
        self.datas = data
  
        self.n_point = n_point
        self.num_cls = num_cls
        self.isAug = isAug
        self.data_path = data_path
        self.max_cls = 15
        self.x = []
        self.y = []
        self.ohe = []

        for i in range(len(self.datas)):
            with open(self.data_path+self.datas[i], 'rb') as f:
                pc_data = np.load(f)
                label = np.load(f)
            f.close()

            if pc_data.shape[0] != self.n_point:
                continue

            pc_data = self.normalize_pc_min_max(pc_data)
            
            if self.isAug:
                pc_data = rotate_point_cloud_x(pc_data)
                pc_data = rotate_point_cloud_y(pc_data)
                pc_data = rotate_point_cloud_z(pc_data)
                pc_data = jitter(pc_data)
                pc_data = shift(pc_data, 0.1)

            y_ohe = np.zeros((len(label), self.max_cls+1), dtype=np.float32)
            y_ohe[np.arange(len(label)), np.array(label)] = 1

            self.x.append(pc_data)
            self.y.append(label)
            self.ohe.append(y_ohe)

    def normalize_pc(self,points):
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance
        return points
    
    def normalize_pc_min_max(self,points):
        points -= points.min(axis=0)
        points /= points.max(axis=0)
        return points
    
    def __getitem__(self, index):

        pc_data = self.x[index]
        label = self.y[index]
        y_ohe = self.ohe[index]

        pc_data = torch.from_numpy(pc_data)
        label = torch.from_numpy(label)
        y_ohe = torch.from_numpy(y_ohe)

        return pc_data, label, y_ohe

    def __len__(self):
        return len(self.x)
    
if __name__ == "__main__":
    train_data_path = "/Volumes/Data/pointnet/" + "part_teeth/train_low_npy/"
    all_train = collect_npy(train_data_path)

    train_data = Obj_dataset(2048, all_train, 15, data_path=train_data_path, isAug=False)

    train_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True
        )
    
    for _ in range(10):
        dataiter = iter(train_loader)   #迭代器
        inputs,labels = next(dataiter)

        new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(inputs,
                                                                          None,
                                                                          512,
                                                                          0.2,
                                                                          32)
        print(grouped_xyz.size())
        # print(labels)

