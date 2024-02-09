import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import random
import fpsample
import pandas as pd
import trimesh
from .aug import shift_y, shift_x, high_low, drop_teeth,rotate_point_cloud_x , rotate_point_cloud_y, rotate_point_cloud_z

def collect_data(path):
    
    data_csv = pd.read_csv(path)
    labels = data_csv["label"].to_list()
    stls = data_csv["obj"].to_list()

    print('number of label :', len(labels))
    print('number of 3D obj :', len(stls))

    return stls, labels

class Obj_dataset(Dataset):
    def __init__(self, n_point, x, y, num_cls, isAug):

        self.X_data = x
        self.y_data = y
        self.n_point = n_point
        self.num_cls = num_cls
        self.isAug = isAug
        self.sub_point = 500
        
        self.xs = []
        self.ys = []

        if isAug:
            sampler_times = 5
        else:
            sampler_times = 1

        for index in range(len(self.X_data)):
            mesh = trimesh.load(self.X_data[index])
            mesh_points = np.asarray(mesh.vertices, dtype=np.float32)
            
            f = open(self.y_data[index])
            label = json.load(f)["instances"]

            for _ in  range(sampler_times):
                idx = self.random_sampler(label)
                labeled = [label[i] for i in idx]

                points = mesh_points[idx]

                # if self.isAug:
                #     points = rotate_point_cloud_x(points)
                #     points = rotate_point_cloud_y(points)
                #     points = rotate_point_cloud_y(points)

                points = self.normalize_pc(points)
                vertices_tensor = torch.from_numpy(points)
                target = torch.tensor(labeled, dtype=torch.long)
                self.xs.append(vertices_tensor)
                self.ys.append(target)


    def random_sampler(self, total):
        cls_dict = {i:[] for i in range(self.num_cls )}
        for idx in range(len(total)):
            cls_dict[total[idx]].append(idx)
        sampler = []

        for i in range(self.num_cls ):
            if len(cls_dict[i]) == 0:
                c = 0
            else:
                c = i
            if len(cls_dict[i]) < self.sub_point:
                subsample = np.random.choice(cls_dict[c], self.sub_point, replace=True)
            else:
                subsample = np.random.choice(cls_dict[c], self.sub_point, replace=False)
            sampler +=  list(subsample)

        random.shuffle(sampler)
        return sampler

    def normalize_pc(self,points):
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance
        return points
    
    def __getitem__(self, index):
        return self.xs[index],self.ys[index]

    def __len__(self):
        return len(self.ys)
    

if __name__ == "__main__":
    all_x, all_y = collect_data("data/data_path_14.csv")

    train_data = Obj_dataset(7500, all_x[:24], all_y[:24], 15)
    train_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True
        )
    for _ in range(10):
        dataiter = iter(train_loader)   #迭代器
        inputs,labels = next(dataiter)

        print(inputs.size())
        print(labels)

