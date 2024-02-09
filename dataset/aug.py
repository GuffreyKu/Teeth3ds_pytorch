import numpy as np
import random

def shift(pc:np.array, shift_range):
    rint = random.randint(0,1)
    N, C = pc.shape
    if rint:
        shifts = np.random.uniform(-shift_range, shift_range, (1, C))
        pc += shifts    
        return pc
    else:
        return pc

def jitter(pc, sigma=0.01, clip=0.05):
    rint = random.randint(0,1)
    N, C = pc.shape
    assert(clip > 0)
    if rint:
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
        pc += jittered_data
        return pc
    else:
        return pc


def shift_y(pc:np.array, gt:np.array):
    rint = random.randint(0,1)
    if rint:
        unilabel = np.unique(gt)

        if 0 in unilabel:
            unilabel = np.delete(unilabel, [0])

        target_c = np.random.choice(unilabel)
        
        unilabel = np.append(unilabel, [0])

        aug_data = []
        aug_gt = []
        for c in unilabel:
            idx = np.where(gt==c)[0]
            sub_pc = pc[idx]
            sub_gt = gt[idx]

            if c == target_c:
                sub_pc[:,1] -= np.random.uniform(-0.05, 0.05)
            aug_data.append(sub_pc)
            aug_gt.append(sub_gt)

        aug_data = np.concatenate(aug_data)
        aug_gt = np.concatenate(aug_gt)

            
        return aug_data, aug_gt
    else:
        return pc, gt

def shift_x(pc:np.array, gt:np.array):
    rint = random.randint(0,1)
    if rint:
        unilabel = np.unique(gt)

        if 0 in unilabel:
            unilabel = np.delete(unilabel, [0])

        target_c = np.random.choice(unilabel)
        
        unilabel = np.append(unilabel, [0])

        aug_data = []
        aug_gt = []
        for c in unilabel:
            idx = np.where(gt==c)[0]
            sub_pc = pc[idx]
            sub_gt = gt[idx]

            if c == target_c:
                sub_pc[:,0] -= np.random.uniform(-0.05, 0.05)
            aug_data.append(sub_pc)
            aug_gt.append(sub_gt)

        aug_data = np.concatenate(aug_data)
        aug_gt = np.concatenate(aug_gt)

        return aug_data, aug_gt
    else:
        return pc, gt
    
def high_low(pc:np.array, gt:np.array):
    rint = random.randint(0,1)
    if rint:
        unilabel = np.unique(gt)

        if 0 in unilabel:
            unilabel = np.delete(unilabel, [0])

        target_c = np.random.choice(unilabel)
        
        unilabel = np.append(unilabel, [0])

        aug_data = []
        aug_gt = []
        for c in unilabel:
            idx = np.where(gt==c)[0]
            sub_pc = pc[idx]
            sub_gt = gt[idx]

            if c == target_c:
                sub_pc[:,2] -= np.random.uniform(-0.05, 0.05)
            aug_data.append(sub_pc)
            aug_gt.append(sub_gt)

        aug_data = np.concatenate(aug_data)
        aug_gt = np.concatenate(aug_gt)

        return aug_data, aug_gt
    else:
        return pc, gt

def rotate_point_cloud_x(pc):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rint = random.randint(0,1)
    if rint:
        
        rotation_angle = np.random.uniform(-np.pi, np.pi)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval, cosval]])
        pc = np.dot(pc.reshape((-1, 3)), rotation_matrix)
        pc = np.asarray(pc, dtype=np.float32)

        return pc
    else:
        return pc
    
def rotate_point_cloud_y(pc):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rint = random.randint(0,1)
    if rint:
        
        rotation_angle = np.random.uniform(-np.pi, np.pi)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        pc = np.dot(pc.reshape((-1, 3)), rotation_matrix)
        pc = np.asarray(pc, dtype=np.float32)

        return pc
    else:
        return pc


def rotate_point_cloud_z(pc):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original  point clouds
        Return:
          Nx3 array, rotated  point clouds
    """
    rint = random.randint(0,1)
    if rint:
        rotation_angle = np.random.uniform(-np.pi, np.pi)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])

        pc= np.dot(pc.reshape((-1, 3)), rotation_matrix)
        pc = np.asarray(pc, dtype=np.float32)
        return pc
    else:
        return pc