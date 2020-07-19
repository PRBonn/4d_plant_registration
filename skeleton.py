""" Skeleton class for creating, accessing different attributes for a skelton. 
"""
import numpy as np

class Skeleton():
  def __init__(self, XYZ = None, edges = None , labels = None):
    self._XYZ = XYZ if XYZ is not None else np.empty((0,3))
    self._edges = edges if edges is not None else []
    self._labels = labels if labels is not None else []
    # initialize A matrix
    self.__compute_adjacency_matrix__()
  
     
  def add_vertex(self, vertex):
    """ Adds a vertex/node to the skeleton graph
    """
    self._XYZ = np.vstack((self._XYZ, vertex))
    #update A matrix
    self.__compute_adjacency_matrix__()

  def add_edge(self, vertex1_id, vertex2_id):
    """ Adds an edge to the skeleton graph
    """
    edge = np.array([vertex1_id, vertex2_id], dtype=np.uint8)
    self._edges.append(edge)
    # update A matrix
    self.__compute_adjacency_matrix__()

  def add_label(self, label):
    """ Assigns a label to a vertex/node in the skeleton graph
    """
    self._labels.append(label)

  def __compute_adjacency_matrix__(self):
    """ Computes an adjecency matrix from the edge list. 
    """
    num_vertices =self._XYZ.shape[0]
    self.A = np.zeros([num_vertices, num_vertices], dtype=np.uint8)
    if num_vertices > 0:
      for e in self._edges:
        if e[0] != e[1]:
          self.A[e[0], e[1]] = 1
          self.A[e[1], e[0]] = 1

  def get_sequence(self):
    """ Computes a sequence along the skeleton in a depth-first manner.
    """     
    if self._XYZ.shape[0] > 0:
      root_idx = np.argmin(self.XYZ[:, 2])
      seq = self.__graph_depth_first_traversal__(root_idx)
    else:
      seq = None
    return seq

  def __graph_depth_first_traversal__(self, root_idx, seq=None, old_root_idx=-1):
    """ Recursive function to traverse the skeleton graph in depth first manner.
    """
    if seq == None:
      seq = []
    seq.append(root_idx)
    for i in range(self.A.shape[0]):
      if i != old_root_idx and self.A[root_idx, i] == 1:
        seq = self.__graph_depth_first_traversal__(i, seq, root_idx)
    return seq

  @property
  def XYZ(self):
    return self._XYZ
  
  @property
  def edges(self):
    return self._edges

  @property
  def labels(self):
    return self._labels
      
  @property
  def node_count(self):
    return self._XYZ.shape[0]

  @property
  def edge_count(self):
    return len(self._edges)

  @classmethod
  def read_graph(cls, filename, matlab_type = False):
    """ Read a graph from a text file as a skeleton
    """
    # read all vertices and edges
    with open(filename, "r") as file:
      vertices = []
      edges = []
      labels = []
      for line in file:
        data = line.split()
        if data[0] == 'v':
          v = np.array([float(data[1]), float(data[2]), float(data[3])])
          vertices.append(v)
          if len(data) > 4:
              l = int(float(data[4]))
              labels.append(l)
        if data[0] == 'e':
          if matlab_type:
              e = np.array([float(data[1])-1, float(data[2])-1], dtype=np.uint8)    
          else:
              e = np.array([float(data[1]), float(data[2])], dtype=np.uint8)
          edges.append(e)
    
    XYZ = np.stack(vertices)
    return cls(XYZ, edges, labels)

  @classmethod
  def copy_skeleton(cls, S):
    """ Make a copy of the skeleton
    """
    # read all vertices and edges
    return cls(S._XYZ.copy(), S._edges.copy(), S._labels.copy())