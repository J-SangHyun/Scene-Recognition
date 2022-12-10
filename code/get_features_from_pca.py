import numpy as np

def get_features_from_pca(feat_num, feature):

    """
    This function loads 'vocab_*.npy' file and
	returns dimension-reduced vocab into 2D or 3D.

    :param feat_num: 2 when we want 2D plot, 3 when we want 3D plot
	:param feature: 'HoG' or 'SIFT'

    :return: an N x feat_num matrix
    """

    vocab = np.load(f'vocab_{feature}.npy')

    # Your code here. You should also change the return value.
    N, D = vocab.shape
    mu = np.mean(vocab, axis=0)
    for i in range(N):
        vocab[i] -= mu
    cov_matrix = np.cov(vocab.T)
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)
    eig_sort_idx = eig_values.argsort()[::-1]
    eig_values, eig_vectors = eig_values[eig_sort_idx], eig_vectors[:, eig_sort_idx]
    eig_vectors = eig_vectors[:, :feat_num]
    pca_features = vocab @ eig_vectors
    return pca_features

    # return np.zeros((vocab.shape[0], feat_num), dtype=np.float32)
