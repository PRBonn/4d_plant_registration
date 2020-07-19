#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a demo for the skeleton matching (correspondence estimation) between a pair of skeletons.
"""
import skeleton as skel
import skeleton_matching as skm
import numpy as np
import matplotlib.pyplot as plt
import visualize as vis

# Load data
species = 'maize'
day1 = '03-13'
day2 = '03-14'
skel_path = '../data/{}/{}.graph.txt'
S1_maize = skel.Skeleton.read_graph(skel_path.format(species, day1))
S2_maize = skel.Skeleton.read_graph(skel_path.format(species, day2))

# Perform matching
params = {'weight_e': 0.01, 'match_ends_to_ends': True,  'use_labels' : True, 'label_penalty' : 1, 'debug': False}
corres = skm.skeleton_matching(S1_maize, S2_maize, params)
print("Estimated correspondences: \n", corres)
 
# visualize results
fh = plt.figure()
vis.plot_skeleton(fh, S1_maize,'b')
vis.plot_skeleton(fh, S2_maize,'r')
vis.plot_skeleton_correspondences(fh, S1_maize, S2_maize, corres) 
plt.title("Estimated correspondences between skeletons")