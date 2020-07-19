#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a demo for iterative non-rigid registration procedure between a pair of skeletons.
"""

import skeleton as skel
import numpy as np
import matplotlib.pyplot as plt
import skeleton_matching as skm
import non_rigid_registration as nrr
from iterative_registration import iterative_registration
import visualize as vis

# %% load skeleton data
species = 'maize'
day1 = '03-13'
day2 = '03-14'
skel_path = '../data/{}/{}.graph.txt'
S1 = skel.Skeleton.read_graph(skel_path.format(species, day1))
S2 = skel.Skeleton.read_graph(skel_path.format(species, day2))

# visualize input data
fh1 = plt.figure()
vis.plot_skeleton(fh1, S1, 'b')
vis.plot_skeleton(fh1, S2, 'r')
plt.show()

# %% compute non-rigid registration params

# set matching params
match_params = {'weight_e': 0.01,
                'match_ends_to_ends': True,
                'use_labels' : False,
                'label_penalty' : 1,
                'debug': False}

# set registration params
reg_params = {'num_iter': 20,
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

# iterative procedure params
params = {'num_iter' : 5,
          'visualize': True,
          'match_params': match_params,
          'reg_params': reg_params}

# call register function
T12, corres = iterative_registration(S1, S2, params)

# %% Apply registration params to skeleton
S2_hat = nrr.apply_registration_params_to_skeleton(S1, T12)

# %% visualize registration results
fh = plt.figure()
vis.plot_skeleton(fh, S1,'b');
vis.plot_skeleton(fh, S2_hat,'k');
vis.plot_skeleton(fh, S2,'r');
vis.plot_skeleton_correspondences(fh, S2_hat, S2, corres)
plt.title("Skeleton registration results")

