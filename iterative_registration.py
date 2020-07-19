#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import skeleton as skel
import skeleton_matching as skm
import non_rigid_registration as nrr
import matplotlib.pyplot as plt
import visualize as vis

def iterative_registration(S1, S2, params):
  """
  Iterative procedure for non-rigid registration of skeleton graphs.

  Parameters
  ----------
  S1, S2 : Skeleton Class
    Two skeletons for which we compute the non-rigid registration params
  params : Dictionary
    num_iter :     Maximum number of iterations used in the procedure 
                   default: 10
    match_params : parameters used for correspondence estimation step 
                   See skeleton_matching module for details.
    reg_params :   parameters used for correspondence estimation step 
                   See non_rigid_registration module for details.

  Returns
  -------
  T12 : list of 4x4 numpy arrays 
    Affine transformation corresponding to each node in S1
  corres : numpy array (Mx2)
    correspondence between two skeleton nodes

  """
  # default params
  if 'num_iter' not in params:
    params['num_iter'] = 10
  
  # params for matching and registration module
  match_params = params['match_params']
  reg_params = params['reg_params']
        
  # Initialize solution
  m = S1.XYZ.shape[0]
  R_init = []
  t_init = []
  for i in range(m):
    R_init.append(np.eye(3))
    t_init.append(np.zeros((3,1)))
  reg_params['R_init'] = R_init
  reg_params['t_init'] = t_init
  
  # initialize  
  S1_transformed =  skel.Skeleton.copy_skeleton(S1)
  old_corres = -np.ones([m,2])

  
  # perform matching and deformation in a loop
  for i in range(params['num_iter']):
    print('-------------------- Global Iteration {} --------------------------'.format(i))
    
    # find correspondences
    corres = skm.skeleton_matching(S1_transformed, S2, match_params)
          
    if params['visualize']:
      fh_vis = plt.figure()
      vis.plot_skeleton(fh_vis, S1,'b');
      vis.plot_skeleton(fh_vis, S2,'r');
      vis.plot_skeleton_correspondences(fh_vis, S1, S2, corres)
      plt.title("Iteration {}: # of Matches = {} ".format(i, corres.shape[0]))
      plt.show()
      
    # find registration params
    T12 = nrr.register_skeleton(S1, S2, corres, reg_params)
    
    # apply registration params to skeleton S1
    S1_transformed = nrr.apply_registration_params_to_skeleton(S1, T12)
    
    # compute registration error for skeleton nodes
    err = nrr.compute_skeleton_registration_error(S1, S2, corres, T12)
    print('Registration error for skeleton nodes = ', err)

    # use results as approximates for next iteration
    for j in range(m):
      reg_params['R_init'][j] = T12[j][0:3,0:3]
      reg_params['t_init'][j] = np.reshape(T12[j][0:3,3], (3,1))
    
    
    # Is there any change in correspondences ? If no, stop
    if corres.shape[0] == old_corres.shape[0] and np.array_equal(np.sort(corres, axis=0), np.sort(old_corres, axis=0)):
      break;
    old_corres = corres.copy()    
    
     # visualize registration results
    if params['visualize']:
      fh_vis = plt.figure()  
      vis.plot_skeleton(fh_vis, S1,'b');
      vis.plot_skeleton(fh_vis, S2,'r');
      vis.plot_skeleton(fh_vis, S1_transformed,'k');
      vis.plot_skeleton_correspondences(fh_vis, S1_transformed, S2, corres)
      plt.title("Iteration {}: Registration ".format(i))
      plt.show()         

  return T12, corres
