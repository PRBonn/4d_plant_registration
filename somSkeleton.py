import minisom
import numpy as np
import scipy.spatial as spatial
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
import copy


def trainSom(data, x, y, epochs, verbose=False):
    """
    trains a som given the input data

    :param data: input, must be [Nx3]
    :param x: number of nodes in the x direction
    :param y: number of nodes in the y direction
    :param epochs: number of training iterations
    :param verbose: if True prints information during training
    :return: result of the som training
    """
    som = minisom.MiniSom(x, y,  3, sigma=0.5, learning_rate=0.5, random_seed=1)
    som.random_weights_init(data)
    som.train_random(data, epochs, verbose=verbose)
    winmap = som.win_map(data)
    return winmap


def somSkeleton(data, x=3, y=1, epochs=20000):
    """
    trains a som and decodes the winning units into skeleton nodes

    :param data: input, must be [Nx3]
    :param x: number of nodes in the x direction
    :param y: number of nodes in the y direction
    :param epochs: number of training iterations
    :return: the nodes that will form the skeleton
    """
    winmap = trainSom(data, x, y, epochs)
    skeleton = []
    for w in winmap:
        w_array = np.asarray(winmap[w])
        w_mean = np.mean(w_array, axis=0)
        skeleton.append(w_mean)

    return skeleton


def computeSkelPoints(cluster, label):
    """
    computes the number of nodes that will form the skeleton

    :param cluster: input data
    :param label: id of the class, 0 for stem, else leaf instances
    :return: number of skeleton nodes in the given set
    """
    if label == 0:
        skel_points = len(cluster)//600
        skel_points = 5 if skel_points < 5 else skel_points
        skel_points = 50 if skel_points > 50 else skel_points
    else:
        skel_points = len(str(len(cluster))) - 1
        skel_points = 2 if skel_points < 2 else skel_points
        skel_points = 6 if skel_points > 6 else skel_points
    return skel_points


def getSkeleton(organs):
    """
    for each organ, returns the nodes of the skeleton

    :param organs: the point cloud segmented into stem and leaves, each list in organs represents a class
    :return: the nodes of the skeleton, each list in skeletons represents a class
    """
    skeletons = []
    for i, o in enumerate(organs):

        if i == 0:
            skel_points = computeSkelPoints(o, i)
            skeleton = somSkeleton(o, x=skel_points, y=1)
            skeletons.append(skeleton)

        else:
            skel_points = computeSkelPoints(o, i)
            skeleton = somSkeleton(o, x=skel_points, y=1)
            skeletons.append(skeleton)

    return skeletons


# ------------------------------------------------------------------------------------------------------------------

def decodeAdjMat(adj_matrix, candidates):
    """
    from adjacency matrix to list of edges

    :param adj_matrix: adjacency matrix encoding the skeleton-like graph
    :param candidates: xyz-coordinates of the nodes in the skeleton
    :return: a list of edges, representing the skeleton-like graph
    """
    refined_edges = []
    n = len(candidates)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] != 0:
                refined_edges.append([candidates[i], candidates[j]])
    return refined_edges


def connect2Stem(leaf, stem):
    """
    given a leaf and the stem, finds the closest stem point to any of the leaf point

    :param leaf: list of points of the current leaf instance
    :param stem: list of points classified as stem
    :return: the closest stem point to any of the leaf point
    """
    candidates = []

    for i, l in enumerate(leaf):
        distances = [np.linalg.norm((l - s)) for s in stem]
        candidates.append(distances)

    stem_candidate = np.where(candidates == np.min(candidates))
    idx = stem_candidate[1][0]
    return stem[idx]


def buildLeafGraph(leaf):
    """
    computes the edges, given the nodes in a leaf

    :param leaf: list of nodes of the current leaf instance
    :return: a list of edges, representing the skeleton-like graph
    """
    adj_matrix = np.zeros((len(leaf), len(leaf)))
    for i, p_i in enumerate(leaf):
        for j, p_j in enumerate(leaf):
            adj_matrix[i, j] = np.linalg.norm((p_j-p_i))

    spanning_tree = minimum_spanning_tree(adj_matrix)
    adj_matrix = spanning_tree.toarray()

    edges = decodeAdjMat(adj_matrix, leaf)

    return np.asarray(edges)


def buildStemGraph(stem, points):
    """
    computes the edges, given the nodes of the stem

    :param stem: list of nodes of the stem
    :param points: all the points classified as stem
    :return: a list of edges, representing the skeleton-like graph
    """
    edges = []
    adj_matrix = np.zeros((len(stem), len(stem)))

    kdt = spatial.cKDTree(points)
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(stem)
    for i, s in enumerate(stem):
        distances, indices = nbrs.kneighbors([s])
        for j in range(len(distances[0])):
            if 0 < distances[0][j] < 30:
                e = [s, stem[indices[0][j]]]
                midpoint = np.mean(e, axis=0)
                nbr = kdt.query_ball_point(midpoint, 5)
                if len(nbr) > 0:
                    edges.append(e)
                    idx = indices[0][j]
                    adj_matrix[i, idx] = distances[0][j]

    spanning_tree = minimum_spanning_tree(adj_matrix)
    adj_matrix = spanning_tree.toarray()

    edges = []  # decoding spanning tree
    for i in range(len(stem)):
        for j in range(len(stem)):
            if adj_matrix[i, j] != 0:
                edges.append([stem[i], stem[j]])

    return np.asarray(edges)


def getGraph(skeletons, pcd):
    """
    computes the edges for each organ

    :param skeletons: the nodes of the skeleton, each list in skeletons represents a class
    :param pcd: the complete plant point cloud
    :return: a list of edges, representing the skeleton-like graph
    """

    graph = []
    for i, s in enumerate(skeletons):
        if i == 0:
            stem = buildStemGraph(s, pcd)
            graph.append(np.asarray(stem))

        else:
            branch_node = connect2Stem(s, skeletons[0])
            current_leaf = copy.deepcopy(s)
            current_leaf.append(branch_node)
            leaf = buildLeafGraph(current_leaf)
            graph.append(np.asarray(leaf))

    return graph