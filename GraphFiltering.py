import numpy as np
from HeterophilicGraph import normalize_matrix, remove_self_loop

def LowPassFilter(X, S, k1, p=0.5):
    I = np.eye(S.shape[0])
    S = S + I
    S = normalize_matrix(S)
    L_S = I - S
    H_low = X.copy()
    for i in range(k1):
        H_low = (I - p * L_S).dot(H_low)
    return H_low

def HighPassFilter(X, S_bar, k2, p=0.5):
    I = np.eye(S_bar.shape[0])
    S_bar_ = remove_self_loop(S_bar)
    S_bar_ = normalize_matrix(S_bar_)
    L_S_bar = I - S_bar_

    H_high = X.copy()
    for i in range(k2):
        H_high = p * L_S_bar.dot(H_high)

    return H_high

# * To reproduce the results in the paper, fine-tune the parameters mu and k1, k2.
def MixedFilter(X, S, S_bar, k1, k2, mu=0.01) :
    H_low = LowPassFilter(X, S, k1)
    H_high = HighPassFilter(X, S_bar, k2)
    H = (1 - mu) * H_low + mu * H_high
    return H


if __name__ == '__main__':
    pass


