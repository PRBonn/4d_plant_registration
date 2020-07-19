#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a demo for estimating the non-rigid registration parameters between a pair of skeletons.
"""
import skeleton as skel
import numpy as np
import matplotlib.pyplot as plt
import skeleton_matching as skm
import non_rigid_registration as nrr
import visualize as vis

# %% load skeleton data and correpondences (matching results)
species = 'maize'
day1 = '03-13'
day2 = '03-14'
skel_path = '../data/{}/{}.graph.txt'
corres_path = '../data/{}/{}-{}.corres.txt'
S1 = skel.Skeleton.read_graph(skel_path.format(species, day1))
S2 = skel.Skeleton.read_graph(skel_path.format(species, day2))
corres = np.loadtxt(corres_path.format(species, day1, day2), dtype = np.int32)

# visualize input data
fh1 = plt.figure()
vis.plot_skeleton(fh1, S1, 'b')
vis.plot_skeleton(fh1, S2, 'r')
vis.plot_skeleton_correspondences(fh1, S1, S2, corres)
plt.title('Skeltons with correspondences.')
plt.show()

# %% compute non-rigid registration params
# set the params
params = {'num_iter': 20,
          'w_rot' : 100,
          'w_reg' : 100,
          'w_corresp' : 1,
          'w_fix' : 1,
          'fix_idx' : [],
          'R_fix' : [np.eye(3)],
          't_fix' : [np.zeros((3,1))],
          'use_robust_kernel' : True,
          'robust_kernel_type' : 'cauchy',
          'robust_kernel_param' : 2,
          'debug' : False}

# call register function
T12 = nrr.register_skeleton(S1, S2, corres, params)

# %% Apply registration params to skeleton
S2_hat = nrr.apply_registration_params_to_skeleton(S1, T12)


# %% visualize registration results
fh = plt.figure()
vis.plot_skeleton(fh, S1,'b');
vis.plot_skeleton(fh, S2_hat,'k');
vis.plot_skeleton(fh, S2,'r');
vis.plot_skeleton_correspondences(fh, S2_hat, S2, corres)
plt.title("Skeleton registration results.")
