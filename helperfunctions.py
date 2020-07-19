#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# helper functions 
M = lambda R,t: np.vstack((np.hstack((R,t)),np.array([[0, 0, 0, 1]])))
hom2euc = lambda x :  x[:-1,:]/x[-1,:]
euc2hom = lambda x : np.vstack((x, np.ones((1,x.shape[1]))))

def jacobian(err_fct, eps, idx, *args ):

  # convert tuple to list to allow for modifying args
  args = list(args)
  I = len(idx)
  
  # initialize Jacobian
  f = err_fct(*args)
  Jac = np.empty((len(f),0), dtype=np.float64)
  
  for i in range(I):
    J = len(args[idx[i]])
    
    for j in range(J):
      
      # left
      args[idx[i]][j] = args[idx[i]][j] - eps
      fl = err_fct(*args) 

      # right
      args[idx[i]][j] = args[idx[i]][j] + 2*eps
      fr = err_fct(*args) 
      
      # reset
      args[idx[i]][j] = args[idx[i]][j] - eps
      
      # Jacobian
      dj = (fr - fl)/(2*eps)
      Jac = np.hstack((Jac, dj))     
     
  return Jac
