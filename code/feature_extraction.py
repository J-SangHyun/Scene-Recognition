import cv2
import numpy as np


def feature_extraction(img, feature):
    """
    This function computes defined feature (HoG, SIFT) descriptors of the target image.

    :param img: a height x width x channels matrix,
    :param feature: name of image feature representation.

    :return: a number of grid points x feature_size matrix.
    """

    if feature == 'HoG':
        # HoG parameters
        win_size = (32, 32)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        deriv_aperture = 1
        win_sigma = 4
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64

        # Your code here. You should also change the return value.
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma,
                                histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)
        hog_feature = hog.compute(img).reshape(-1, 36)
        return hog_feature

        # return np.zeros((1500, 36), dtype=np.float32)
        # `.shape[0]` do not have to be (and may not) 1500,
        # but `.shape[1]` should be 36.

    elif feature == 'SIFT':

        # Your code here. You should also change the return value.
        sift = cv2.SIFT_create()
        grid_size = 20
        keypoints = [cv2.KeyPoint(x, y, grid_size)
                     for x in range(0, img.shape[0], grid_size)
                     for y in range(0, img.shape[1], grid_size)]
        _, sift_feature = sift.compute(img, keypoints)
        return sift_feature

        # return np.zeros((1500, 128), dtype=np.float32)
        # `.shape[0]` do not have to be (and may not) 1500,
        # but `.shape[1]` should be 128.



