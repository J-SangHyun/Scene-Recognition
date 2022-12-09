import numpy as np
from sklearn import svm


def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats:
        an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels:
        an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats:
        an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.
    :param kernel_type:
        the name of a kernel type. 'linear' or 'RBF'.

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)

    # Your code here. You should also change the return value.
    # svm.SVC(), svm.LinearSVC(), svm.SVC.fit(), svm.SVC.decision_function() from scikit-learn.
    svms = [svm.LinearSVC() if kernel_type == 'linear' else svm.SVC() for _ in range(len(categories))]
    scores = np.zeros((len(categories), len(test_image_feats)))
    for i in range(len(categories)):
        label = np.zeros((len(train_labels),))
        label[train_labels == categories[i]] = 1
        label[train_labels != categories[i]] = -1
        svms[i].fit(X=train_image_feats, y=label)
        scores[i] = svms[i].decision_function(X=test_image_feats)

    scores = scores.T
    prediction = np.array([categories[np.argmax(scores[i])] for i in range(len(test_image_feats))])
    return prediction

    # return np.array([categories[0]] * 1500)
