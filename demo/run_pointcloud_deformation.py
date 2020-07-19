#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a demo for pointcloud deformation using the skeleton registration parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

import skeleton as skel
import skeleton_matching as skm
import pointcloud as pcd
import visualize as vis

# %% Load data
species = 'maize'
day1 = '03-13'
day2 = '03-14'
skel_path = '../data/{}/{}.graph.txt'
pc_path = '../data/{}/{}.xyz'
corres_path = '../data/{}/{}-{}.corres.txt'
reg_path = '../data/{}/{}-{}.reg.npy'
S1 = skel.Skeleton.read_graph(skel_path.format(species, day1))
S2 = skel.Skeleton.read_graph(skel_path.format(species, day2))
P1 = pcd.load_pointcloud(pc_path.format(species, day1))
P2 = pcd.load_pointcloud(pc_path.format(species, day2))
corres = np.loadtxt(corres_path.format(species, day1, day2), dtype = np.int32)
T12 = np.load(reg_path.format(species, day1, day2))  

# deform complete pointcloud
P1_ds = pcd.downsample_pointcloud(P1, 3)
P2_ds = pcd.downsample_pointcloud(P2, 3)
P1_deformed = pcd.deform_pointcloud(P1_ds, T12, corres, S1, S2)

# visualize results
fh = plt.figure()
vis.plot_skeleton(fh, S1,'b')
vis.plot_skeleton(fh, S2,'r')
vis.plot_pointcloud(fh, P2,'r')
vis.plot_pointcloud(fh, P1_deformed,'b')
vis.plot_skeleton_correspondences(fh, S1, S2, corres, 'g')