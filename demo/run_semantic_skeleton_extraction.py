#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a demo for extracting the skeleton structure from the plant point cloud.
"""
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from joblib import load
import matplotlib.pyplot as plt
import visualize as vis
import somSkeleton as som
import skeleton as skel
import utils

# setting input point cloud
species = 'maize'
day = '03-13'
model_path = '../models/{}_svm.joblib'

# loading point cloud and downsample
path = '../data/{}/{}.ply'
P = o3d.io.read_point_cloud(path.format(species, day))
P = P.voxel_down_sample(voxel_size=0.5)

# compute FPFH features
fpfh, xyz = utils.computeHistograms(P)
features = np.column_stack((fpfh, xyz))
fh = plt.figure()
vis.plot_pointcloud(fh, xyz, color='gray')
plt.title('Input point cloud scan')

# load pre-trained svm model
clf = load(model_path.format(species))

# perform stem vs leaf classification
predict = clf.predict(features)
data = np.column_stack((xyz, predict))
stem = data[np.where(data[:, -1] == 0)]
leaf = data[np.where(data[:, -1] == 1)]

# perform clustering
dbscan = DBSCAN(min_samples=2, p=0).fit(leaf[:, :-1])
leaves, labels = utils.refineClustering(leaf[:, :-1], dbscan.labels_)
organs = utils.prepare4Skel(stem, leaves) 

# visualize different organs
fh = plt.figure()
vis.plot_semantic_pointcloud(fh, organs)
plt.title('Sematic classification of plant organs')

# perform skeletonization
cwise_skeleton_nodes = som.getSkeleton(organs)
graph = som.getGraph(cwise_skeleton_nodes, xyz)

# convert this graph to skeleton class
S = utils.convert_to_skeleton_class(cwise_skeleton_nodes, graph)

# plot skeleton
fh = plt.figure()
vis.plot_skeleton(fh, S)
plt.title('Skeleton structure of the plant')