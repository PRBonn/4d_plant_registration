import numpy as np
import matplotlib.pyplot as plt
from dijkstar import Graph, find_path
import skeleton as skel

def skeleton_matching(S1, S2, params):
  """
  Computes the correspondences given for a skeleton pair using a HMM formulation.

  Parameters
  ----------
  S1, S2 : Skeleton Class
    Two skeletons for which we compute the correspondences
  params : Dictionary
    Can be an empty dict (All params have some default values)
    
    weight_e: Used in builing the emissions matrix E.It weighss the difference between the
              skeleton point vs. the difference in the degree of a graph node.
              Set to low number if you want more matches end-points to end-points
              default: 10.0
    match_ends_to_ends: if false, an endpoint to a middlepoint gets equal weights for E
                        as middlepoint to middlepoint.
                        default:false
    use_labels:         If true, then labels are used for the emmision cost.
                        Labels must be given in S.labels for every node.
                        default: false
    label_penalty:      penalty for emmision if labels are not the same.
                        default: 1 (1 means: same cost as node degree with a difference of 1)
    debug:              show plots and other info (true or false)
                        default: false
  Returns
  -------
  corres : numpy array [Mx2]
    column 0 has node ids from S1, and column 1 has corresponding node ids from S2.

  """  
  print("Computing matches.")
  # set default params is not provided
  # in emissions E  : weights the difference between the
  # skeleton point vs. the difference in the degree of a graph node
  if 'weight_e' not in params:
    params['weight_e'] = 10.0

  # use semantic labels
  if 'use_labels' not in params:
    params['use_labels'] = False
  
  # apply label penalty or not
  if 'label_penalty' not in params:
    params['label_penalty'] = False
  
  # show debug msgs/vis
  if 'debug' not in params:
    params['debug'] = False
    

  # define HMM S1->S2
  print("HMM: Computing Transition and Emission probabilities")
  T1, E1, statenames1 = define_skeleton_matching_hmm(S1,S2, params);
  
  # Transform T and E to probability
  to_prob = lambda x : 1/x
  T1 = to_prob(T1)    
  E1 = to_prob(E1)
  
  # compute correspondence pairs using viterbi
  V = np.array(S1.get_sequence())
  best_seq = viterbi(V, T1, E1, statenames1)
  corres = get_correspondences_from_seq(best_seq)
  
  # remove all matchings to virtual 'nothing' node
  ind_remove = np.where(corres[:,1]==-1)
  corres = np.delete(corres, ind_remove, 0)

  # post process
  corres = remove_double_matches_in_skeleton_pair(S1, S2, corres)
  
  # visualize matching results
  if params['debug']:
    fh_debug = plt.figure()
    skel.plot_skeleton(fh_debug, S1,'b')
    skel.plot_skeleton(fh_debug, S2,'r')
    plot_skeleton_correspondences(fh_debug, S1, S2, corres) 

  return corres
  
    

def define_skeleton_matching_hmm(S1, S2, params):
  """
  Define cost matrices for Hidden Markov Model for matching skeletons

  Parameters
  ----------
  S1, S2 :  Skeleton Class
            Two skeletons for which we compute the correspondences
  params :  Dictionary
            see skeleton_matching function for details

  Returns
  -------
  T : numpy array 
      Defines the cost from one pair to another pair
  E : numpy array
      emmision cost matrixd efines the cost for observing one pair as a match
  statenames : list 
               names for each state used in the HMM (all correspondence pairs)  

  """
  # define statenames
  statenames = define_statenames(S1, S2)
 
  # Precompute geodesic distances
  GD1, NBR1  = compute_geodesic_distance_on_skeleton(S1)
  GD2, NBR2  = compute_geodesic_distance_on_skeleton(S2)
   
  # Precompute euclidean distances between all pairs
  ED = compute_euclidean_distance_between_skeletons(S1, S2)

  # compute transition matrix
  T = compute_transition_matrix(S1, S2, GD1, NBR1, GD2, NBR2, ED)      

  # compute emission matrix 
  E = compute_emission_matrix(S1, S2, ED, params)
  
  return T, E, statenames

def define_statenames(S1, S2):
  # number of nodes in each skeleton
  N = S1.XYZ.shape[0]
  M = S2.XYZ.shape[0]

  statenames = []
  # starting state
  statenames.append([-2, -2])
  for n1 in range(N):
    for m1 in range(M):
        statenames.append([n1, m1])
  for n1 in range(N):
    statenames.append([n1, -1])    

  return statenames

def get_correspondences_from_seq(seq):
  N = len(seq)
  corres = np.zeros((N,2), dtype=np.int64)
  for i in range(N):
    corres[i,0] = seq[i][0]
    corres[i,1] = seq[i][1]
    
  return corres

def compute_geodesic_distance_on_skeleton(S):
  G = Graph(undirected=True)
  edges = S.edges
  for e in edges:
    edge_length =  euclidean_distance(S.XYZ[e[0],:], S.XYZ[e[1],:])
    G.add_edge(e[0], e[1], edge_length)
  
  N = S.XYZ.shape[0]
  GD = np.zeros([N,N])
  NBR = np.zeros([N,N])
  for i in range(N):
    for j in range(i+1,N):           
      gdist, nbr = geodesic_distance_no_branches(G, S.A, i, j)
      GD[i,j]=gdist
      GD[j,i]=gdist
      NBR[i,j]=nbr
      NBR[j,i]=nbr
  
  return GD, NBR

 
def geodesic_distance_no_branches(G, A, i, j):
  path = find_path(G, i, j)
  gdist = path.total_cost
  
  nbr = 0
  for i in range(1, len(path.nodes)):
    degree = sum(A[path.nodes[i],:])
    nbr = nbr + (degree-2) # A normal inner node has degree 2, thus additional branches are degree-2

  return gdist , nbr

def mean_distance_direct_neighbors(S, GD):
  N = S.XYZ.shape[0]
  sumd=0
  no=0
  for i in range(N):
    for j in range(N):
      if S.A[i,j]:
        sumd = sumd + GD[i,j]
        no=no+1
     
  mean_dist = sumd/no

  return mean_dist


def euclidean_distance(x1, x2):
  dist = np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 + (x1[2]-x2[2])**2)
  return dist

def compute_euclidean_distance_between_skeletons(S1, S2):
  N = S1.XYZ.shape[0]
  M = S2.XYZ.shape[0]
  
  D = np.zeros([N,M])
  for n1 in range(N):
    for m1 in range(M):
      D[n1,m1] =  euclidean_distance(S1.XYZ[n1,:], S2.XYZ[m1,:])

  return D
  
def remove_double_matches_in_skeleton_pair(S1, S2, corres):
  """
  Remove double matches of two skeletons and keeps only the one with the smallest distance.

  Parameters
  ----------
  S1, S2 : Skeleton Class
           Two skeletons for which we compute the correspondences
  corres : numpy array (Mx2)
           correspondence between two skeleton nodes pontetially with one-to-many matches

  Returns
  -------
  corres: numpy array
          one-to-one correspondences between the skeleton nodes.

  """
  num_corres = corres.shape[0]
  distances = np.zeros((num_corres,1))
  for i in range(num_corres):
    distances[i] = euclidean_distance(S1.XYZ[corres[i,0],:], S2.XYZ[corres[i,1],:])

  # remove repeated corres 1 -> 2
  corres12, counts12 = np.unique(corres[:,0], return_counts = True)
  ind_remove12 = []
  for i in range(len(corres12)):
    if counts12[i] > 1:
      ind_repeat = np.argwhere(corres[:,0] == corres12[i])
      dist_repeat = distances[ind_repeat]
      ind_ = np.argsort(dist_repeat)[1:]
      ind_remove12.append(ind_repeat[ind_])
  corres = np.delete(corres, ind_remove12, axis=0)
  distances = np.delete(distances, ind_remove12, axis=0)    
  
  # remove repeated corres 2 -> 1
  corres21, counts21 = np.unique(corres[:,1], return_counts = True)
  ind_remove21 = []
  for i in range(len(corres21)):
    if counts21[i] > 1:
      ind_repeat = np.argwhere(corres[:,1] == corres21[i]).flatten()
      dist_repeat = distances[ind_repeat].flatten()
      ind_ = np.argsort(dist_repeat)[1:]
      ind_remove21.append(ind_repeat[ind_])
  corres = np.delete(corres, ind_remove21, axis=0)
  distances = np.delete(distances, ind_remove21, axis=0)    
        
  return corres


def compute_transition_matrix(S1, S2, GD1, NBR1, GD2, NBR2, ED):
  
  N = S1.XYZ.shape[0]
  M = S2.XYZ.shape[0]
  T= 1e6*np.ones([N*M +N, N*M +N], dtype=np.float64)

  mean_dist_S1 = mean_distance_direct_neighbors(S1, GD1)
  max_cost_normal_pairs=0

  # normal pair to normal pair:
  # (n1,m1) first pair, (n2,m2) second pair
  max_cost_normal_pairs = 0
  for n1 in range(N):
    for m1 in range(M):
      for n2 in range(N):
        for m2 in range(M):        
          # Avoid going in different directions on the skeleton. Then the
          # geodesic difference can be small, but actually one would assume a
          # large cost
          v_inS1 =  S1.XYZ[n2,:] - S1.XYZ[n1,:]
          v_inS2 = S2.XYZ[m2,:] - S2.XYZ[m1,:]
         
          # angle between vectors smaller 90 degrees
          if np.dot(v_inS1, v_inS2) >= 0:                    
            # geodesic distance and difference in number of branches along the way
            g1 = GD1[n1,n2]
            g2 = GD2[m1,m2]
            br1= NBR1[n1,n2]
            br2= NBR2[m1,m2]
            v = np.abs(br1-br2)*np.max(GD1) + np.abs(g1-g2) + mean_dist_S1           
            T[n1*M+m1, n2*M+m2] = v

            if v > max_cost_normal_pairs:
               max_cost_normal_pairs = v

  # Main diagonal should be large
  for i in range(N*M):
    T[i,i] = max_cost_normal_pairs

  # Normal pair -> not present
  for n1 in range(N):
    for m1 in range(M):
      for n2 in range(N):
        if n2 != n1:
          T[n1*M+m1 , N*M + n2] = np.max(GD1)/2 

  # Not present -> normal pair
  for n2 in range(N):
    for m1 in range(M):
      for n1 in range(N):
        T[N*M + n2, n1*M+m1] = max_cost_normal_pairs

  # Not present -> not present
  S1_seq = S1.get_sequence()
  for n1 in range(N):
    for n2 in range(N):
      pos_n1 = S1_seq.index(n1) if n1 in S1_seq else -1 
      pos_n2 = S1_seq.index(n2) if n2 in S1_seq else -1
      if pos_n2 > pos_n1:
        v = GD1[n1, n2]
        T[N*M + n1, N*M + n2] = v + mean_dist_S1 + np.min(ED[n1,:])
  
  # Add starting state (which will not be reported in hmmviterbi)           
  sizeT = N*M +N;
  T1 = np.hstack((1e6*np.ones((1,1)), np.ones((1,N*M)), 1e6*np.ones((1,N))))     
  T2 = np.hstack((1e6*np.ones((sizeT, 1)), T))
  T = np.vstack( (T1, T2))
  
  return T 

def compute_emission_matrix(S1, S2, ED, params):
     
  # emmision matrix (state to sequence)
  # Here: degree difference + euclidean difference inside a pair
  N = S1.XYZ.shape[0]
  M = S2.XYZ.shape[0]
  E = 1e05*np.ones((N*M +N, N))
   
  for n1 in range(N):
    for m1 in range(M):
       
      degree1= float(np.sum(S1.A[n1,:]))
      degree2= float(np.sum(S2.A[m1,:]))
              
      # Do not penalize end node against middle node
      if not params['match_ends_to_ends'] and ((degree1==1 and degree2==2) or (degree1==2 and degree2==1)):
        E[n1*M+m1, n1] = params['weight_e'] * (ED[n1,m1] + 10e-10)
      else:
        E[n1*M+m1, n1]= np.abs(degree1-degree2) + params['weight_e'] * (ED[n1,m1] + 10e-10)


  # Add penalty if labels are not consistent
  if params['use_labels']:
    for n1 in range(N):
      for m1 in range(M):
        if S1.labels[n1] != S2.labels[m1]:
          E[n1*M+m1, n1] = E[n1*M+m1, n1] + params['label_penalty']

  # No match
  for n1 in range(N):
    # Take the  best
    I = np.argsort(E[0:N*M, n1])
    E[N*M +n1, n1] = E[I[0], n1]
      
  # Add starting state (which will not be reported in hmmviterbi)
  E = np.vstack((1e10*np.ones((1,N)),E))

  return E


def viterbi(V, T, E, StateNames):
  """
  This function computes a sequence given observations, transition and emission prob. using the Viterbi algorithm.
  

  Parameters
  ----------
  V : numpy array (Mx1)
      Observations 
  T : numpy array (NxN)
      Transition probabilities 
  E : numpy array (NxM)
      Emission probabilities 
  StateNames : list
                ames for each state used in the HMM

  Returns
  -------
  S :  list 
      Best sequence of hidden states
     
  """
  M = V.shape[0]
  N = T.shape[0]

  omega = np.zeros((M, N))
  omega[0, :] = np.log(E[:, V[0]])

  prev = np.zeros((M - 1, N))

  for t in range(1, M):
    for j in range(N):
      # Same as Forward Probability
      probability = omega[t - 1] + np.log(T[:, j]) + np.log(E[j, V[t]])

      # This is our most probable state given previous state at time t (1)
      prev[t-1, j] = np.argmax(probability)

      # This is the probability of the most probable state (2)
      omega[t, j] = np.max(probability)

    # Path Array
    S_ = np.zeros(M)

    # Find the most probable last hidden state
    last_state = np.argmax(omega[M - 1, :])

    S_[0] = last_state

  backtrack_index = 1
  for i in range(M-2,-1, -1):
    S_[backtrack_index] = prev[i, int(last_state)]
    last_state = prev[i, int(last_state)]
    backtrack_index += 1

  # Flip the path array since we were backtracking
  S_ = np.flip(S_, axis=0)

  # Convert numeric values to actual hidden states
  S = []
  for s in S_:
    S.append(StateNames[int(s)])
      
  return S
