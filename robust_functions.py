#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def loss_l2(x, c=2):
  L = 0.5 * x * x / (c * c)
  D = x
  W = np.ones(len(x))
  
  return L, D, W

def loss_huber(x, c=2):
  L = np.zeros(len(x))
  D = np.zeros(len(x))
  W = np.zeros(len(x))

  for i in range(len(x)):   
    if np.abs(x[i]) <= c:
        L[i] = x[i] * x[i] / 2
        D[i] = x[i]
        W[i] = 1
    else:
        L[i] = c * (np.abs(x[i]) - c / 2)
        D[i] = c * np.sign(x[i])
        W[i] = 1 / np.abs(x[i])
    
  return L, D, W


def loss_cauchy(x, c=2):
  L = c * c / 2 * np.log(1 + x * x / c)
  D = x / (1 + (x / c) * (x / c))
  W = 1 / (1 + (x / c) * (x / c))
   
  return L, D, W
  
def loss_geman_mcclure(x, c=2):  
  L = (0.5*c*x**2)/(c + x**2)
  D = (c**2*x)/((c + x**2)**2)
  W = (c**2)/((c + x**2)**2)

  return L, D, W
    