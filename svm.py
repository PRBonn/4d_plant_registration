from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
import numpy as np
import open3d as o3d
from joblib import dump

import utils


def sampleClf(pcd, clf, labels):
    """
    computes the label of each point in the downsampled point cloud

    :param pcd: downsampled point cloud
    :param clf: original point cloud
    :param labels: labels for each point in clf
    :return: labels for each point in pcd
    """
    nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(clf)
    _, indices = nbrs.kneighbors(pcd)

    nn_labels = labels[indices]
    new_labels = np.max(nn_labels, axis=1)
    return new_labels


def getTrainDataset(paths):
    """
    loads data and labels, computes fpfh

    :param paths: list of paths to train data
    :return: fpfh, xyz coordinates, labels
    """

    labels = np.array([])
    histograms = np.array([])
    xyz = np.array([])

    for p in paths:
        # loading data, downsampling it and removing ground points
        data = np.loadtxt(p)
        print('loaded: {}'.format(p))

        data = data[np.where(data[:, 3] != 0)]
        points = data[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size=1)

        # converting instance labels to stem/leaf
        classes = data[:, 3]
        classes[classes == 1] = 0
        classes[classes > 1] = 1

        classes = sampleClf(np.asarray(pcd.points), points, classes)

        # computing fpfh
        fpfh, points = utils.computeHistograms(pcd)
        print('computed fpfh')

        # concatenating in a single variable
        labels = np.concatenate([labels, classes]) if labels.size else classes
        histograms = np.concatenate([histograms, fpfh]) if histograms.size else fpfh
        xyz = np.concatenate([xyz, points]) if xyz.size else points

    return histograms, xyz, labels


if __name__ == '__main__':

    species = ''
    train_paths = []
    model_path = ''

    histograms, xyz, labels = getTrainDataset(train_paths)
    features = np.column_stack((histograms, xyz))

    svm = SVC(gamma='scale', decision_function_shape='ovo').fit(features, labels)
    dump(svm, model_path.format(species))

