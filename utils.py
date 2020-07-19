import numpy as np
import open3d as o3d
from sklearn.neighbors import KNeighborsClassifier

import skeleton as skel


def removeOutliers(array, std_ratio=0.5):
    """
    remove outliers from a point cloud

    :param array: input cloud
    :param std_ratio: Standard deviation ratio
    :return: inlier set, outlier set
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(array))
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=300, std_ratio=std_ratio)

    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)

    inliers = np.asarray(inlier_cloud.points)
    ouliers = np.asarray(outlier_cloud.points)
    return inliers, ouliers


def computeHistograms(pcd):
    """
    computes Fast Point Features Histograms (FPFH)

    :param pcd: input cloud
    :return: FPFH, xyz coordinates
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.5, max_nn=30))
    pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamRadius(radius=5))
    return np.asarray(pcd_fpfh.data).T, np.asarray(pcd.points)


def prepare4Skel(stem, leaves):
    """
    for each organ, removes outliers

    :param stem: points classified as stem
    :param leaves: points classified as leaves, each list is a different instance
    :return: list of classes, stem is always at index 0
    """
    stem, _ = removeOutliers(stem[:, :-1])
    organs = [stem]
    for l in leaves:
        l, _ = removeOutliers(np.asarray(l))
        organs.append(l)
    return organs


def refineClustering(xyz, labels):
    """
    discards small clusters and assigns those to bigger leaves

    :param xyz: list of points in the cloud
    :param labels: corresponding labels
    :return: refined clusters, once discarded small ones
    """
    leaves, keep_labels = getLeaves(xyz, labels)

    fixed = []
    fixed_labels = []
    for i in range(len(xyz)):
        if labels[i] in keep_labels:
            fixed.append(xyz[i])
            fixed_labels.append((labels[i]))

    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
    neigh.fit(np.asarray(fixed), np.asarray(fixed_labels))

    for i in range(len(xyz)):
        if labels[i] not in keep_labels:
            l = neigh.predict([xyz[i]])
            labels[i] = l[0]

    leaves, _ = getLeaves(xyz, labels)
    return leaves, labels


def getLeaves(xyz, labels):
    """
    discard small clusters

    :param xyz: list of points in the cloud
    :param labels: corresponding labels
    :return: the points and label that survived the threshold
    """
    keep_labels = []
    threshold = 100
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        if c > threshold:
            if u != -1:
                keep_labels.append(u)

    leaves = []
    for label in keep_labels:
        points = []
        for i in range(len(xyz)):
            if label == labels[i]:
                points.append(xyz[i])

        leaves.append(points)
    return np.asarray(leaves), keep_labels


def convertEdgesFormat(points, graph):
    """
    convert edges from xyz coordinates to indexes

    :param points: list of nodes in the skeleton
    :param graph: list of edges, as xyz coordinates
    :return: list of edges, as indexes
    """
    edges = []
    for g in graph:
        for point in g:
            edges.append(point)

    ids = []
    for e in edges:
        edge_ids = [np.inf, np.inf]
        for i, p in enumerate(points):
            if np.equal(e[0], p[:-1]).all():
                edge_ids[0] = i
            if np.equal(e[1], p[:-1]).all():
                edge_ids[1] = i
        ids.append(edge_ids)
    return ids


def saveSkeleton(skeletons, graph, filename):
    """
    save graph as txt file:

        v x1 y1 z1 l1
              .
              .
              .
        v xn yn zn ln
        e i1 j1
           .
           .
           .
        e in jn

    :param skeletons: skeleton nodes
    :param graph: skeleton edges
    :param filename: path to save the file
    """
    points = []
    for label, s in enumerate(skeletons):
        for p in s:
            points.append(np.append(p, label))

    edges = convertEdgesFormat(points, graph)

    string = ""
    for p in points:
        string += 'v {} {} {} {}\n'.format(*p)

    for e in edges:
        string += 'e {} {}\n'.format(*e)

    file = open(filename, 'w')
    file.write(string)
    file.close()

    
def convert_to_skeleton_class(cnodes, graph):
    """ converts som skeleton to general skeleton type """
    points = []
    for label, s in enumerate(cnodes):
      for p in s:           
        points.append(np.append(p, label))
    edges = convertEdgesFormat(points, graph)

    # Put in the skeleton structure 
    S = skel.Skeleton()
    for i in range(len(points)):
      p = points[i]
      V = np.array(p[0:3], dtype=np.float)
      S.add_vertex(V)
      L = int(p[3])
      S.add_label(L)
    
    for e in edges:
      S.add_edge(e[0], e[1])

    return S