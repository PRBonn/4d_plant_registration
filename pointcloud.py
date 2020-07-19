#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pathlib

import helperfunctions as hf

def deform_pointcloud(P1, T12, corres, S1, S2):
  """
  Deform a pointcloud using deformation parameters computed by registering 
  corresponding skeletons.
  ALGORITHM USED HERE:
  - The first neighbour n1 is the nearest node
  - The second neighbour n2 is a node connected to n1. If n1 has more
    than one neigbour (the standard case), then the one on the 'right' side
    is chosen. 'right' side means that the projection lies really on the
    edge between n1 and n2.

  Parameters
  ----------
  P1 : numpy array (Nx3)
    XYZ coordinates of the first point cloud
  T12 : list of 4x4 numpy arrays 
    Affine transformation corresponding to each node in S1
  corres : numpy array (Mx2)
    correspondence between two skeleton nodes
  S1, S2 : Skeleton Class
    Two skeletons for which we compute the non-rigid registration params
  Returns
  -------
  P1_deformed : numpy array (Nx3)
    P1 after undergoing deformation params given in T12 

  """
  # Find nearest skeleton node
  num_points = P1.shape[0]
  # Matrix of nearby info:
  # col 1: node index of nearest
  # col 2: distance to nearest
  nearest= np.zeros((num_points,2))
  for i in range(num_points):
    diff = S1.XYZ- P1[i,:] 
    dist = np.sqrt(np.sum(diff**2, axis=1))
    indices = np.argsort(dist)
    nearest[i,:] = np.array([[indices[0], dist[indices[0]]]])
  
  # perform deformation
  P1_deformed = P1.copy()
  for i in range(num_points):
    n1 = int(nearest[i,0])
    n2_candidates = S1.A[n1,:].nonzero()[0]
    # in between:
    # >0 weight f2 for node n2
    # -1 == is on the other side of n1
    # -2 == is on the other side of n2
    in_between = np.zeros(len(n2_candidates))
    for j in range(len(n2_candidates)):
      n2 = n2_candidates[j]
      line_direction = S1.XYZ[n2,:] - S1.XYZ[n1,:] 
      # projection relative to n1
      line_projection = (((P1[i,:]-S1.XYZ[n1,:]) @ line_direction) / (line_direction@line_direction) ) * line_direction
      # absolute projection
      projection = S1.XYZ[n1,:] + line_projection
      
      f2 = np.linalg.norm(line_projection)/np.linalg.norm(line_direction)
      in_between[j] = f2
      
      if np.dot(line_projection, line_direction) < 0:
        # The projected point does not lie in between the two nodes, it is on
        # the other side of nearest1
        # ==> use another node or only nearest1 trafo
        in_between[j] = -1
      else:
        if f2 > 1:
          # The projected point does not lie in between the two nodes, it is on
          # the other side of nearest2
          # ==> use another node or only nearest2 trafo
          in_between[j] = -2
           
    # find best candidate
    J = (in_between>=0).nonzero()[0]
    if len(J) > 0:
      # in-between node
      if len(J) == 1:
        n2 = n2_candidates[J[0]]
        f2 = in_between[J[0]]
      else:
        # more than one in between: can happen for nodes with degree>=3:
        # choose the n2, where the projection is nearest to the point
        dmin = np.inf
        for j in range(len(J)):
          # projection absolute
          projection = S1.XYZ[n1,:] + ( ( (P1[i,:]-S1.XYZ[n1,:]) @ line_direction) / (line_direction @ line_direction) ) * line_direction
          d = np.linalg.norm(projection - P1[i,:])
          
          if d < dmin:
            dmin = d
            n2 = n2_candidates[J[j]]
            f2 = in_between[J[j]]
    else:
      n2 = n2_candidates[0]
      if in_between[0] == -1:
        f2 = 0
      else:
        f2 = 1
    
    # apply transformation to the point
    f1 = 1 - f2
    T1 = T12[n1]
    T2 = T12[n2]
    Pn1 = hf.hom2euc(T1 @ hf.euc2hom(np.reshape(P1[i,:],(3,1))))
    Pn2 = hf.hom2euc(T2 @ hf.euc2hom(np.reshape(P1[i,:],(3,1))))
    P1_deformed[i,:] = (f1*Pn1 + f2*Pn2).flatten()
  
  return P1_deformed

def load_pointcloud(pc_file):
  file_ext = pathlib.Path(pc_file).suffix
  if file_ext == '.txt' or file_ext == '.xyz':
    P = np.loadtxt(pc_file)
  elif file_ext == '.npy' or file_ext == '.npz':
    P = np.load(pc_file)
  else:
    print('Pointcloud file type not supported.')
  
  return P

def downsample_pointcloud(P, ds):
  P_ds = P[::ds,:].copy()
  return P_ds



