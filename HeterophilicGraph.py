
import numpy as np
import torch


def TopK(arr, k):

    arr = torch.from_numpy(arr)
    vas, ids = torch.topk(arr, k)
    k_v, k_ids = torch.min(vas, dim=1)
    k_v = k_v.numpy()
    arr = arr.numpy()
    arr = arr.T
    arr[arr < k_v] = 0

    return arr.T


def remove_self_loop(A):

    A_new = np.triu(A, 1) + np.tril(A, -1)
        
    return A_new

def normalize_matrix(A, eps=1e-12):

    D = np.sum(A, axis=1) + eps

    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0
    D[np.isnan(D)] = 0
    D = np.diagflat(D)
    A = D.dot(A).dot(D)

    return A

#* this is core code for construct H
def construct_heterophilic_graph(K=None, A=None, steps=5, Knn=5) :
    one_matrix = np.ones_like(K)
    I = np.eye(K.shape[0])
    A_new = remove_self_loop(A)
    M = np.linalg.matrix_power(A_new + I, steps)
    M[M > 0] = 1
    
    K_bar = one_matrix - K
    K_bar = TopK(K_bar, Knn)
    A_bar = one_matrix - M

    A_bar = remove_self_loop(A_bar)

    K_hat = np.multiply(A_bar, K_bar)

    return K_hat