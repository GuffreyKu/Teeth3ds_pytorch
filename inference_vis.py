import torch

import numpy as np
import fpsample
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_point = 4096
num_cls = 15

pointnet = torch.load("saved_model/pointnet_glm.pt", map_location="cpu")

def normalize_pc(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance
    return points

def normalize_pc_min_max(points):
    points -= points.min(axis=0)
    points /= points.max(axis=0)
    return points

def get_boundary(points, labeled):
    distances = cdist(points, points, metric='euclidean')

    nearest = np.where(distances < 0.01818301530559778)
    boundary = []
    for i in range(nearest[0].shape[0]):
        xi = nearest[0][i]
        yi = nearest[1][i]
        label_x = labeled[xi]
        label_y = labeled[yi]

        if label_x != label_y:
            boundary.append(xi)
            boundary.append(yi)
    
    final_label = []
    for p in range(len(labeled)):
        if p in boundary:
            final_label.append(17)
        else:
            final_label.append(0)

    print(len(set(boundary)))
    return points, final_label

if __name__ == "__main__":
    import pandas as pd
    from scipy.spatial.distance import cdist

    ## for keke data
    # mesh = trimesh.load("test/2-1I17_FDI_LowerJaw.stl")
    # points = np.asarray(mesh.vertices, dtype=np.float32)

    ## for opensource data
    data = pd.read_csv("/Volumes/Data/pointnet/part_teeth/lower/00OMSZGW_lower_3.csv")
    points = data[['x', 'y', 'z']].values
    label = data['label'].to_list()

    idx = fpsample.bucket_fps_kdline_sampling(points, n_point, h=9)
    idx = [int(x) for x in idx]

    points = points[idx].astype(np.float32)
    labeled = [label[i] for i in idx]

    points = normalize_pc_min_max(points)

    adj = cdist(points, points, metric='euclidean')
    S1 = np.zeros([points.shape[0], points.shape[0]], dtype='float32')
    S1[adj<0.1] = 1.0

    S1 = torch.from_numpy(S1)
    S1 = S1.unsqueeze(0)
    
    vertices_tensor = torch.from_numpy(points)
    vertices_tensor = vertices_tensor.unsqueeze(0)

    pointnet.eval()
    with torch.no_grad():

        S1 = S1.to(device)
        vertices_tensor = vertices_tensor.to(device)

        outputs = pointnet(vertices_tensor, S1)
    
    pred = torch.softmax(outputs, dim=2).argmax(dim=2)
    pred = pred.numpy(force=True)

    print(' ---- PD count ----')
    unique, counts = np.unique(pred, return_counts=True)
    print(np.asarray((unique, counts)).T)

    print(' ---- GT count ----')
    unique, counts = np.unique(labeled, return_counts=True)
    print(np.asarray((unique, counts)).T)

    # # Assuming 'points' is your point cloud data
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=pred, marker='.', cmap=plt.cm.Set2)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.show()