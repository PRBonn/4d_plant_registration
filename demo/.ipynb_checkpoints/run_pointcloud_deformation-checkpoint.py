#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import skeleton as skel
import skeleton_matching as skm
import pointcloud as pcd

# %% Load data
P1 = pcd.load_pointcloud('data/P1.txt')
P2 = pcd.load_pointcloud('data/P2.txt')
S1 = skel.Skeleton.read_graph('data/S1.graph.txt')
S2 = skel.Skeleton.read_graph('data/S2.graph.txt')
corres = np.loadtxt('data/corres_S1_S2.txt', dtype = np.int32)
T12 = np.load('data/T12.npy')  

# %% deform point cloud
# downsample pc
P1 = pcd.downsample_pointcloud(P1, 3)
P2 = pcd.downsample_pointcloud(P2, 3)
P1_deformed = pcd.deform_pointcloud(P1, T12, corres, S1, S2)

# %% visualize deformed point cloud
fh = plt.figure()
skel.plot_skeleton(fh, S1,'b')
#pcd.plot_pointcloud(fh, P1,'b')
skel.plot_skeleton(fh, S2,'r')
pcd.plot_pointcloud(fh, P1_deformed)
skm.plot_skeleton_correspondences(fh, S1, S2, corres, 'g')