#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

import skeleton as skel
import skeleton_matching as skm
import robust_functions as rf
import helperfunctions as hf
import visualize as vis

Obs = namedtuple('Observation', 'g q')

def register_skeleton(S1, S2, corres, params):
  """
  This function computes the (non-rigid) registration params between the
  two skeletons. This function computes the normal equation, solves the
  non-linear least squares problem.
  

  Parameters
  ----------
  S1, S2 : Skeleton Class
    Two skeletons for which we compute the non-rigid registration params
  corres : numpy array (Mx2)
    correspondence between two skeleton nodes
  params : Dictionary
    num_iter :  Maximum number of iterations for the optimization routine
                default: 10
    w_rot :     Weight for the rotation matrix constraints
                default: 100,
    w_reg :     Weight for regularization constraints
                default: 100
    w_corresp:  Weight for correspondence constraints 
                default: 1
    w_fix :     Weight for fixed nodes 
                default: 1
    fix_idx :   list of fixed nodes
                default : []
    R_fix :     list of rotation matrices for fixed nodes
                default : [np.eye(3)]
    t_fix :     list of translation vectors for fixed nodes
                default: [np.zeros((3,1))]
    use_robust_kernel :   Use robust kernels for optimization (recommended if corres has outliers)
                          default : True
    robust_kernel_type :  Choose a robust kernel type (huber, cauchy, geman-mcclure)
                          default: 'cauchy'
    robust_kernel_param : scale/outlier parameter for robust kernel 
                          default: 2
    debug :     show debug visualizations + info
                default: False

  Returns
  -------
  T12 : list of 4x4 numpy arrays 
    Affine transformation corresponding to each node in S1

  """

  print('Computing registration params.')
  
   # set default params if not provided
  if 'num_iter' not in params:
    params['num_iter'] = 10
  
  if 'w_rot' not in params:
    params['w_rot'] = 100
  
  if 'w_reg' not in params:
    params['w_reg'] = 100
    
  if 'w_corresp' not in params:
    params['w_corresp'] = 1
  
  if 'w_fix' not in params:
    params['w_fix'] = 1
    
  if 'fix)idx' not in params:
    params['fix_idx'] = []
    
  if 'use_robust_kernel' not in params:
    params['use_robust_kernel'] = False

  if 'robust_kernel_type' not in params:
    params['robust_kernel_type'] = 'cauchy'

  if 'robust_kernel_param' not in params:
    params['robust_kernel_param'] = 2

  if 'debug' not in params:
    params['debug'] = False

  # initialize normal equation
  J_rot, r_rot, J_reg, r_reg, J_corresp, r_corresp, J_fix, r_fix = \
    initialize_normal_equations(S1, corres, params)
  
  # initialze solution
  x = initialize_solution(S1, params)
  
  # initialize weights
  W_rot, W_reg, W_corresp, W_fix = initialize_weight_matrices(\
    params['w_rot'], len(r_rot), params['w_reg'], len(r_reg), \
    params['w_corresp'], len(r_corresp) , params['w_fix'], len(r_fix))

  # # initialize variables in optimization
  m = S1.XYZ.shape[0]
  T12 = [None]*m
  R = [None]*m
  t = [None]*m
  for j in range(m):
    xj = x[12*j:12*(j+1)]
    R[j] = np.reshape(xj[0:9], (3,3))
    t[j] = xj[9:12]
    
  # perform optimization
  if params['debug']:
    fh_debug = plt.figure()
  E_prev = np.inf;
  dx_prev = np.inf;
    
  for i in range(params['num_iter']):
    
    # counters used for different constraints
    jk = 0
    jc = 0
    jf = 0
   
    # compute jacobian  and residual for each constraint types
    for j in range(m):
      # registration params for jth node
      Rj = R[j]
      tj = t[j]    
      
      # constraints from rotation matrix entries
      Jj_rot, rj_rot = compute_rotation_matrix_constraints(Rj)
      J_rot[6*j:6*(j+1), 12*j:12*(j+1)] = Jj_rot
      r_rot[6*j:6*(j+1)] = rj_rot
      
      # constraints from regularization term
      ind  = np.argwhere(S1.A[j,:]==1).flatten()
      for k in range(np.sum(S1.A[j,:])): 
        
        # params
        Rk = R[ind[k]]
        tk = t[ind[k]]
            
        Jj_reg, Jk_reg, r_jk_reg = compute_regularization_constraints(Rj, tj, Rk, tk)
            
        # collect all constraints
        nc = r_jk_reg.shape[0]
        J_reg[nc*jk : nc*(jk+1), 12*j:12*(j+1)] = Jj_reg
        J_reg[nc*jk : nc*(jk+1), ind[k]*12:12*(ind[k]+1)] = Jk_reg
        r_reg[nc*jk : nc*(jk+1)] = r_jk_reg
 
        # increment counter for contraints from neighbouring nodes
        jk = jk+1           
    
      # constraints from correspondences
      if corres.shape[0] > 0:
        ind_C = np.argwhere(corres[:,0] == j).flatten()
        if len(ind_C) > 0:
 
          # observations
          Y = Obs(S1.XYZ[j,:].reshape(3,1), S2.XYZ[corres[ind_C,1],:].reshape(3,1))
          
          # compute constraints
          J_jc_corresp, r_jc_corresp = compute_corresp_constraints(Rj, tj, Y)
                
          # collect all constraints
          nc = r_jc_corresp.shape[0]
          J_corresp[nc*jc:nc*(jc+1), 12*j:12*(j+1)] = J_jc_corresp
          r_corresp[nc*jc:nc*(jc+1)] = r_jc_corresp

          # increment counter for correspondence constraints
          jc = jc + 1
                
   
      # constraints from fixed nodes
      if len(params['fix_idx']) > 0:
        if j in params['fix_idx']:
          ind_f = params['fix_idx'].index(j)
   
          # observations
          R_fix = params['R_fix'][ind_f]
          t_fix = params['t_fix'][ind_f]
                
          # compute fix node constraints
          J_jf_fix, r_jf_fix = compute_fix_node_constraints(Rj, tj, R_fix, t_fix);
          nc = r_jf_fix.shape[0]
                
          J_fix[nc*jf: nc*(jf+1), 12*j:12*(j+1)] = J_jf_fix
          r_fix[nc*jf:nc*(jf+1)] = r_jf_fix
            
          # update counter
          jf = jf + 1
              
    # compute weights and residual using robust kernel
    if params['use_robust_kernel']:
      if params['robust_kernel_type'] == 'huber':
        _, _, W_corresp = rf.loss_huber(r_corresp, params['robust_kernel_param'])
      
      elif params['robust_kernel_type'] == 'cauchy':
        _, _, W_corresp = rf.loss_cauchy(r_corresp, params['robust_kernel_param'])
 
      elif params['robust_kernel_type'] == 'geman_mcclure':
        _, _, W_corresp = rf.loss_geman_mcclure(r_corresp, params['robust_kernel_param'])
    
      else:
        print('Robust kernel not undefined. \n')

      W_corresp = params['w_corresp']*np.diag(W_corresp.flatten())    
      
    # collect all constraints    
    J = np.vstack((J_rot, J_reg, J_corresp, J_fix))
    r = np.vstack((r_rot, r_reg, r_corresp, r_fix))
    
    # construct weight matrix
    W = combine_weight_matrices(W_rot, W_reg, W_corresp, W_fix)
    
    # solve linear system
    A = J.T @ W @ J
    b = J.T @ W @ r
    dx = -np.linalg.solve(A, b)
    
    # Errors
    E_rot = r_rot.T @ W_rot @ r_rot
    E_reg = r_reg.T @ W_reg @ r_reg
    E_corresp = r_corresp.T @ W_corresp @ r_corresp
    E_fix = r_fix.T @ W_fix @ r_fix
    E_total = E_rot + E_reg + E_corresp + E_fix
        
    # print errors
    if params['debug']:
      print("Iteration # ", i)
      print("E_total = ", E_total)
      print("E_rot = ", E_rot)
      print("E_reg = ", E_reg)
      print("E_corresp = ", E_corresp)
      print("E_fix = ", E_fix)
      print("Rank(A) = ", np.linalg.matrix_rank(A))
    
    # update current estimate
    for j in range(m):
      #params
      dx_j = dx[12*j:12*(j+1)]
      R[j] = R[j] + np.reshape(dx_j[0:9], (3, 3), order = 'F')
      t[j] = t[j] + dx_j[9:12]
    
    # collect and return transformation
    for j in range(m):
      T12[j] = hf.M(R[j], t[j])
    
    # apply registration to skeleton for visualization
    if params['debug']:
      # compute registration error
      S2_hat = apply_registration_params_to_skeleton(S1, T12)
      vis.plot_skeleton(fh_debug, S1,'b');
      vis.plot_skeleton(fh_debug, S2,'r');
      vis.plot_skeleton(fh_debug, S2_hat,'k');
      vis.plot_skeleton_correspondences(fh_debug, S2_hat, S2, corres)
      plt.title("Iteration " + str(i))

        
    # exit criteria     
    if np.abs(E_total - E_prev) < 1e-6 or np.abs(np.linalg.norm(dx) - np.linalg.norm(dx_prev)) < 1e-6:
      print("Exiting optimization.")
      print('Total error = ', E_total)
      break
    
    # update last solution
    E_prev = E_total
    dx_prev = dx
    
  return T12

def initialize_normal_equations(S, corres, params):
  """
  This function initailizes J and r matrices for different types of % constraints.

  Parameters
  ----------
  S :       Skeleton Class
            Contains points, adjacency matrix etc related to the skeleton graph.
  corres :  numpy array (Mx2)
            correspondence between two skeleton nodes
  params :  Dictionary
            see description in register_skeleton function

  Returns
  -------
  J_rot : numpy array [6mx12m]
          jacobian for rotation matrix  error
  r_rot : numpy array [6mx1]
          residual for rotation matrix  error
  J_reg : numpy array [12mx12m]
          jacobian for regularization error
  r_reg : numpy array [12mx1] 
          residual for reuglarization error
  J_corres : numpy array [3nCx12m]
             jacobian for correspondence  error
  r_corres : numpy array [3nCx1] 
             residual for correspondence  error
  J_fix : numpy array[12nFx12m]
          jacobian for fix nodes
  r_fix : numpy array [12mFx1] 
          residual for fix nodes 
  """
  
  # get sizes from input 
  m = S.XYZ.shape[0]
  nK = 2*S.edge_count
  nC = corres.shape[0]
  nF = len(params['fix_idx'])
  
  # constraints from individual rotation matrix
  num_rot_cons = 6*m
  J_rot = np.zeros((num_rot_cons, 12*m))
  r_rot = np.zeros((num_rot_cons,1))
  
  # constraints from regularization
  num_reg_cons = 12*nK
  J_reg = np.zeros((num_reg_cons,12*m))
  r_reg = np.zeros((num_reg_cons,1))
  
  # constraints from correspondences
  num_corres_cons = 3*nC;
  J_corres = np.zeros((num_corres_cons,12*m))
  r_corres = np.zeros((num_corres_cons,1))
  
  # constraints from fix nodes
  num_fix_cons = 12*nF
  J_fix = np.zeros((num_fix_cons,12*m))
  r_fix = np.zeros((num_fix_cons,1))
  
  return J_rot, r_rot, J_reg, r_reg, J_corres, r_corres, J_fix, r_fix

def initialize_solution(S, params):
  """
  This function initialzes the soultion either as the zero solution or 
  provided initial transformation.
  Parameters
  ----------
  S : Skeleton Class
      Only used for getting number of unknowns.
  params : Dictionary
    R_init, params.t_init used for initializing solution if
    they are provided. If R_init is a list then a separate approximate is
    assumed for every transformation. Otherwise R_init should be 3x3, t_init 3x1

  Returns
  -------
  x : numpy array [12mx1] 
      initial solution vector as expected by the optimization procedure.

  """
  m = S.XYZ.shape[0]
  x = np.zeros((12*m,1))
  R = [None]*m
  t = [None]*m
  for j in range(m):
      if 'R_init' in params and  't_init' in params:
          if len(params['R_init']) == m:
            R[j] = params['R_init'][j]
            t[j] = params['t_init'][j]
          else:
            R[j] = params['R_init']
            t[j] = params['t_init']
      else:        
          # start from zero solution
          R[j] = np.eye(3);
          t[j] = np.zeros((3,1))
      
      # rearrange in a column vector
      x[12*j:12*(j+1)] = np.vstack((np.reshape(R[j], (9, 1),order='F'),t[j]))
      
  return x

def initialize_weight_matrices(w_rot, n_rot, w_reg, n_reg, w_corresp, n_corresp, w_fix, n_fix):
  """
  This function computes the weight matrices corresponding to each constraint
  given the weights and number of constraints for each type.
  """
  W_rot = np.diag(w_rot*np.ones(n_rot))
  W_reg = np.diag(w_reg*np.ones(n_reg))
  W_corresp = np.diag(w_corresp*np.ones(n_corresp))
  W_fix = np.diag(w_fix*np.ones(n_fix)) 

  return W_rot, W_reg, W_corresp, W_fix 

def combine_weight_matrices(W_rot, W_reg, W_corresp, W_fix):
  """
  This function combines the weight matrices of each constraint type into 
  the combined weight matrix used in the optimization step. 
  """
  # get number of constraints
  n_rot = W_rot.shape[0]
  n_reg = W_reg.shape[0]
  n_corresp = W_corresp.shape[0]
  n_fix = W_fix.shape[0]
  nC = n_rot + n_reg + n_corresp + n_fix 
  
  # combine them to form the big W matrix
  W = np.zeros((nC, nC))
  W[0:n_rot, 0:n_rot] = W_rot
  W[n_rot:n_rot+n_reg, n_rot:n_rot+n_reg] = W_reg
  W[n_rot+n_reg:n_rot+n_reg+n_corresp, n_rot+n_reg:n_rot+n_reg+n_corresp] = W_corresp
  W[n_rot+n_reg+n_corresp:n_rot+n_reg+n_corresp+n_fix, n_rot+n_reg+n_corresp:n_rot+n_reg+n_corresp+n_fix] = W_fix

  return W

def compute_rotation_matrix_constraints(R):
    
  # constraints from rotation matrix entries
  c1 = R[0:3,0].reshape((3,1))
  c2 = R[0:3,1].reshape((3,1))
  c3 = R[0:3,2].reshape((3,1))
  
  # # Jacobian wrt R (1x9), wrt t (1x3)
  r1 = c1.T @ c2
  Jc_r1 = np.hstack((c2.T, c1.T, np.zeros((1,3))))
  Jt_r1 = np.zeros((1,3))
  
  r2 = c1.T @ c3
  Jc_r2 = np.hstack((c3.T, np.zeros((1,3)), c1.T))
  Jt_r2 = np.zeros((1,3))
  
  r3 = c2.T @ c3
  Jc_r3 =  np.hstack((np.zeros((1,3)), c3.T, c2.T))
  Jt_r3 = np.zeros((1,3))
  
  r4 = c1.T @ c1 -1
  Jc_r4 = np.hstack((2*c1.T, np.zeros((1,3)), np.zeros((1,3))))
  Jt_r4 = np.zeros((1,3))
  
  r5 = c2.T @ c2 -1
  Jc_r5 = np.hstack((np.zeros((1,3)), 2*c2.T, np.zeros((1,3))))
  Jt_r5 = np.zeros((1,3))
  
  r6 = c3.T @ c3 -1
  Jc_r6 = np.hstack((np.zeros((1,3)), np.zeros((1,3)), 2*c3.T))
  Jt_r6 = np.zeros((1,3))
  
  # J:= 6x12, r:= 6x1
  J  = np.vstack((np.hstack((Jc_r1, Jt_r1)),
                  np.hstack((Jc_r2, Jt_r2)),
                  np.hstack((Jc_r3, Jt_r3)),
                  np.hstack((Jc_r4, Jt_r4)),
                  np.hstack((Jc_r5, Jt_r5)),
                  np.hstack((Jc_r6, Jt_r6))))
      
  r = np.vstack((r1,
                 r2,
                 r3,
                 r4,
                 r5,
                 r6))

  return J, r


def compute_regularization_constraints(Rj, tj, Rk, tk):

  # Transformations
  Tj = hf.M(Rj, tj)
  Tk = hf.M(Rk, tk)
  
  # residual (12x1)
  r_eye = Tj @ np.linalg.inv(Tk)
  r =  np.vstack((np.reshape(r_eye[0:3,0:3]- np.eye(3), (9, 1), order='F'), np.reshape(r_eye[0:3,3],(3,1), order='F')))
  
  # jacobian
  x1 =  np.vstack((np.reshape(Rj, (9, 1)), tj))
  x2 =  np.vstack((np.reshape(Rk, (9, 1)), tk))
  Jj = hf.jacobian(residual_reg, 1e-6, [0], x1, x2)
  Jk = hf.jacobian(residual_reg, 1e-6, [1], x1, x2)
  
  return Jj, Jk, r

def compute_corresp_constraints(R, t, Y):
  # residual 3x1
  r =  R @ Y.g + t - Y.q

  # JR=(3x9), Jt=(3x3)
  JR = np.hstack((Y.g[0]*np.eye(3), Y.g[1]*np.eye(3), Y.g[2]*np.eye(3)))
  Jt = np.eye(3)
  J  = np.hstack((JR, Jt))
 
  return J, r 
  
def compute_fix_node_constraints(R, t, R_fix, t_fix):
  # error (12x1), Jc (12x9), Jt( 12x3)
  r = np.vstack((np.reshape(R, (9, 1), order='F'), t)) -np.vstack((np.reshape(R_fix, (9, 1), order='F'), t_fix))
  
  # Note: J is constant.
  Jc = np.vstack((np.eye(9), np.zeros((3,9))))
  Jt = np.vstack((np.zeros((9,3)), np.eye(3)))
  J = np.hstack((Jc, Jt))

  return J, r

def residual_reg(x1, x2):

  # Reshape into Rotation and translation 
  Rj = np.reshape(x1[0:9],(3,3), order='F')
  tj = x1[9:12]
  Rk = np.reshape(x2[0:9],(3,3), order='F')
  tk = x2[9:12]
  
  Tj = hf.M(Rj, tj)
  Tk = hf.M(Rk, tk)
          
  # residual (12x1)
  r_eye = Tj @ np.linalg.inv(Tk)
  r =  np.vstack((np.reshape(r_eye[0:3,0:3]- np.eye(3), (9, 1), order='F'), np.reshape(r_eye[0:3,3],(3,1), order='F')))
    
  return r

def apply_registration_params_to_skeleton(S, T):
  S_hat = skel.Skeleton.copy_skeleton(S)
  m = S.XYZ.shape[0]
  for j in range(m):
    S_hat.XYZ[j,:] = hf.hom2euc(T[j] @ hf.euc2hom(np.reshape(S.XYZ[j,:], (3,1)))).T
 
  return S_hat

def compute_skeleton_registration_error(S1, S2, corres, T12):
  S1_hat = apply_registration_params_to_skeleton(S1, T12)
  err = np.linalg.norm(S1_hat.XYZ[corres[:,0] ,:] - S2.XYZ[corres[:,1] ,:])    
  return err  