import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction


def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_*.npy' exists and
    contains an vocab size x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """

    vocab = np.load(f'vocab_{feature}.npy')

    vocab_size = vocab.shape[0]

    # Your code here. You should also change the return value.
    N = len(image_paths)
    pyramid_size = sum([4 ** level for level in range(max_level+1)])
    feature_reprs = np.zeros((N, pyramid_size * vocab_size))

    for i in range(len(image_paths)):
        img = cv2.imread(image_paths[i])[:, :, ::-1]
        sub_feature_repr = np.zeros((0,))

        for level in range(max_level+1):
            w_l = 2 ** (-max_level + max(level, 1) - 1)
            grid_size = (img.shape[0] // 2 ** level, img.shape[1] // 2 ** level)
            for h in range(0, 2 ** level):
                for w in range(0, 2 ** level):
                    sub_img = img[h*grid_size[0]:(h+1)*grid_size[0], w*grid_size[1]:(w+1)*grid_size[1]]
                    sub_features = feature_extraction(sub_img, feature)
                    sub_dist = pdist(sub_features, vocab)
                    sub_hist = np.zeros((vocab_size,))
                    for j in range(len(sub_features)):
                        sub_hist[np.argmin(sub_dist[j])] += 1.0
                    sub_hist = w_l * sub_hist / np.linalg.norm(sub_hist)
                    sub_feature_repr = np.concatenate([sub_feature_repr, sub_hist], axis=0)
        feature_reprs[i] = sub_feature_repr
    return feature_reprs

    # return np.zeros((1500, 36))
