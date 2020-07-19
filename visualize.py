import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pointcloud(fh, P, color='r'):
  ax = fh.gca(projection='3d')
  ax.scatter3D(P[:,0], P[:,1], P[:,2], '.', color=color, depthshade=False)

  
def plot_semantic_pointcloud(fh, array_list):
    """
    :param array_list: a list of numpy arrays, each array represents a different class
    """
    ax = fh.gca(projection='3d')
    
    for P in array_list:
        P = np.asarray(P)
        ax.scatter3D(P[:,0], P[:,1], P[:,2], '.')

def plot_skeleton(fh, S, color='blue'):    
    """ plots the skeleton graph with nodes and edges """
    # plot vertices
    ax = fh.gca(projection='3d')
    ax.scatter3D(S.XYZ[:,0], S.XYZ[:,1], S.XYZ[:,2], 'o', color=color, depthshade=False)
    
    # plot edges
    N = S.A.shape[0]
    for i in range(N):
        for j in range(N):
            if S.A[i,j] == 1:
                ax.plot3D([S.XYZ[i,0], S.XYZ[j,0]], \
                          [S.XYZ[i,1], S.XYZ[j,1]], \
                          [S.XYZ[i,2], S.XYZ[j,2]], color)
    
    
def plot_skeleton_correspondences(fh, S1, S2, corres, color = 'red'):
 
  # Delete the corresponding to virtual node
  ind_remove = np.where(corres[:,0]==-1)
  corres = np.delete(corres, ind_remove, axis=0)
  ind_remove = np.where(corres[:,1]==-1)
  corres = np.delete(corres, ind_remove, axis=0)

  # plot correspondences    
  ax = fh.gca(projection='3d')
  N = corres.shape[0]
  for i in range(N):
    ax.plot3D([S1.XYZ[corres[i,0],0], S2.XYZ[corres[i,1],0]], \
              [S1.XYZ[corres[i,0],1], S2.XYZ[corres[i,1],1]], \
              [S1.XYZ[corres[i,0],2], S2.XYZ[corres[i,1],2]], color)
