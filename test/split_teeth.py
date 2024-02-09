import numpy as np
import pandas as pd
import os
import json
import trimesh
import matplotlib.pyplot as plt
import random


label_map_lower = {
    0 : 0,
    31 : 1,
    32 : 3,
    33 : 5,
    34 : 7,
    35 : 9,
    36 : 11,
    37 : 13,
    38 : 15,
    41 : 2,
    42 : 4,
    43 : 6,
    44 : 8,
    45 : 10,
    46 : 12,
    47 : 14,
    48 : 16
}

label_map_upper = {
    0 : 0,
    11 : 18,
    12 : 19,
    13 : 20,
    14 : 21,
    15 : 22,
    16 : 23,
    17 : 24,
    18 : 25,
    21 : 26,
    22 : 27,
    23 : 28,
    24 : 29,
    25 : 30,
    26 : 31,
    27 : 32,
    28 : 33
}

def collect_data(mode):
    '''
    mode = "lower" or "upper"
    '''
    labels = []
    stls = []
    
    for rpath in os.listdir("data"):
        if rpath == ".DS_Store":
            continue
        for jpath in os.listdir("data/"+rpath):
            if jpath ==".DS_Store":
                continue
            if mode not in  jpath:
                continue
            for idpath in sorted(os.listdir("data/"+rpath+'/'+jpath)):
                if idpath ==".DS_Store":
                    continue
                for item in os.listdir("data/"+rpath+'/'+jpath + '/' + idpath):
                    if item.split('.')[-1] == 'json':
                        labels.append('data/' + rpath + '/'+ jpath + '/' + idpath + '/' + item)
                    elif item.split('.')[-1] == 'obj':
                        stls.append('data/' + rpath + '/'+ jpath + '/' + idpath  + '/' + item)
 

    print('number of label :', len(labels))
    print('number of 3D obj :', len(stls))

    # print(max(shape), min(shape))

    return stls, labels

def normalize_pc(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance
    return points

def point_split(pc, xr, yr, zr, quadrant):
    '''
    pc: point cloud dataframe, normalize -1 ~ 1
    xr: x over ratio,  -1 ~ 1, if none not split by x
    yr: y over ratio,  -1 ~ 1, if none not split by y
    zr: z over ratio,  -1 ~ 1, if none not split by z
    quadrant:  coordinate quadrant(int 1, 2, 3, 4) 
    '''
    if quadrant == 1:
        if xr is not None:
            pc = pc[pc["x"]>xr]
        if yr is not None:
            pc = pc[pc["y"]>yr]
        if zr is not None:
            pc = pc[pc["z"]>zr]
        return pc
    
    elif quadrant == 2:
        if xr is not None:
            pc = pc[pc["x"]<xr]
        if yr is not None:
            pc = pc[pc["y"]>yr]
        if zr is not None:
            pc = pc[pc["z"]>zr]
        return pc
    
    elif quadrant == 3:
        if xr is not None:
            pc = pc[pc["x"]<xr]
        if yr is not None:
            pc = pc[pc["y"]<yr]
        if zr is not None:
            pc = pc[pc["z"]>zr]
        return pc
    
    elif quadrant == 4:
        if xr is not None:
            pc = pc[pc["x"]>xr]
        if yr is not None:
            pc = pc[pc["y"]<yr]
        if zr is not None:
            pc = pc[pc["z"]>zr]
        return pc

if __name__ == "__main__":
    
    mode = "lower"

    if mode == "lower":
        label_map = label_map_lower
    else:
        label_map = label_map_upper

    all_x, all_y = collect_data(mode)

    for idx in range(len(all_y)):

        name = all_y[idx].split('/')[-1].split('.')[0]

        f = open(all_y[idx])
        label = json.load(f)["labels"]
        label_mapped = [label_map[i] for i in label]

        mesh =  trimesh.load(all_x[idx])
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        vertices = normalize_pc(vertices)

        pc_df = pd.DataFrame()
        pc_df["x"] = vertices[:, 0]
        pc_df["y"] = vertices[:, 1]
        pc_df["z"] = vertices[:, 2]
        pc_df["label"] = label_mapped

        # xr = random.uniform(-0.3, 0.3)
        pc_part_1 = point_split(pc_df, 0, 0 , None, quadrant=1)
        pc_part_2 = point_split(pc_df, 0, 0 , None, quadrant=2)
        pc_part_3 = point_split(pc_df, 0, 0 , None, quadrant=3)
        pc_part_4 = point_split(pc_df, 0, 0 , None, quadrant=4)

        pc_part_1.to_csv("part_teeth/%s/%s_1.csv"%(mode, name), index=False)
        pc_part_2.to_csv("part_teeth/%s/%s_2.csv"%(mode, name), index=False)
        pc_part_3.to_csv("part_teeth/%s/%s_3.csv"%(mode, name), index=False)
        pc_part_4.to_csv("part_teeth/%s/%s_4.csv"%(mode, name), index=False)

        # break

        # Assuming 'points' is your point cloud data
        # x = pc_df_over1["x"]
        # y = pc_df_over1["y"]
        # z = pc_df_over1["z"]

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y, z, c=pc_df_over1["label"], marker='o', cmap=plt.cm.gist_ncar)

        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')

        # ax.set_xlim([-10, 10])
        # ax.set_ylim([-10, 10])
        # ax.set_zlim([-10, 10])
        # plt.show()