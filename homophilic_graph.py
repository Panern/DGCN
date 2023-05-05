
import torch
import torch.nn.functional as F
import numpy as np
import numba
import random
import torch.nn as nn
from tqdm import tqdm
from HeterophilicGraph import normalize_matrix
from sklearn.metrics.pairwise import cosine_similarity

#* this is core code for construct S, part 6
def A_final(S, nbs=50.0) :

    S[S < (1 / nbs)] = 0
    S[S >= (1 / nbs)] = 1
    S[S > 0] = 1
    S = S.T + S
    S[S >= 1] = 1
    S[S < 1] = 0

    return S


#* this is core code for construct S, part 1
@numba.jit
def op_tmp(A, K) :
    L = A.shape[0]
    A_2 = A.dot(A)
    N, M = np.zeros_like(A), np.zeros_like(A)
    for i in range(L) :
        for j in range(L) :
            sum_s_jf_2 = 0
            sum_s_jf_Cf = 0
            for f in range(L) :
                C_f = A_2[i, f] - A[j, f] * A[i, j] - A[i, f]
                if f != j :
                    sum_s_jf_2 += A[j][f] ** 2
                    sum_s_jf_Cf += A[j][f] * C_f
            s_ij_N = 2 * A_2[i, j] - K[i, j] - 2 * sum_s_jf_Cf
            s_ij_M = 4 + 2 * sum_s_jf_2
            N[i, j] = s_ij_N
            M[i, j] = s_ij_M

    return N, M

#* this is core code for construct S, part 2
@numba.jit
def op_S(lb2, N, M):
    S = (N.T + lb2 ) / M.T
    S[S<0] = 0
    return S.T

@numba.jit
def create_W(X, scale=16) :
    
    N = X.shape[0]
    S = np.zeros((N, N))
    
    for i in range(N) :
        for j in range(N) :
            S[i, j] += np.linalg.norm(X[i] - X[j], ord=2) / scale
    return S
    
#* this is core code for construct S, part 3
def compute_NM(dataname="Texas", X=None, A=None) :
    I = np.eye(A.shape[0])
    A = A + I
    A = normalize_matrix(A)
    K = create_W(X)

    N, M = op_tmp(A=A, K=K)

    # np.save('./N_{}.npy'.format(dataname), N)
    # np.save('./M_{}.npy'.format(dataname), M)
    # print(N, M)
    return N, M

#* this is core code for construct S, part 4
def compute_A_bar(N, M, dataname="Texas") :
    # S = np.zeros_like(N)
    lb2 = []
    for i in tqdm(range(N.shape[0])) :
        lb = optimize_lbd2(N[i], M[i])
        lb2.append(lb)
    lb2 = np.array(lb2)
    S = op_S(lb2, N, M)
    # np.save("./S_{}.npy".format(dataname), S)
    return S

#* this is core code for construct S, part 5
#* This is for compute \lambda_2 via gradient descent
#* In fact, this is a approximate solution
#* You can also use the exact solution, but it is too slow
class lambda_2(nn.Module):
    def __init__(self):
        super(lambda_2, self).__init__()
        self.lbd = nn.Parameter(torch.Tensor(1))
        self.reset_parameter()

    def reset_parameter(self):
        self.lbd.data.fill_(1e-3)

    def forward(self, N_i, M_i):
        Lbd = F.relu(self.lbd)
        obj = F.relu((N_i + Lbd)/M_i)
        return obj.sum()


def optimize_lbd2(N_i, M_i, epochs=1000, learning_rate=1e-4, convengence=1e-16):

    N_ii =  torch.from_numpy(N_i)
    seed = 666666
    
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    N_ii = N_ii.cuda()
    M_ii = torch.from_numpy(M_i).cuda()
    model = lambda_2()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model.reset_parameter()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_last = torch.zeros(1).cuda()
    for epoch in range(epochs):
        optimizer.zero_grad()
        obj = model(N_ii, M_ii)
        # print(obj.sum())
        loss = 10 * torch.square(obj - 1)
        if torch.abs_(loss_last - loss) < convengence * loss_last:
            break
        else:
            loss_last = loss.clone()
        loss.backward()
        optimizer.step()

        return F.relu(model.lbd).item()


#* Compute the homophily of the graph
def homophily_v2(A, labels, ignore_negative=False) :
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    if ignore_negative = True, then only compute for edges where nodes both have
        nonnegative class labels (negative class labels are treated as missing
    """
    src_node, targ_node = A.nonzero()
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    if ignore_negative :
        edge_hom = np.mean(matching[labeled_mask])
    else :
        edge_hom = np.mean(matching)
    return edge_hom




if __name__ == '__main__':
    
    #! A simple example on Cora dataset
    from load_data import Swither
    from HeterophilicGraph import  remove_self_loop, construct_heterophilic_graph
    import matplotlib.pyplot as plt
    
    X, A, gnd = Swither[2](dataname="Cora")
   
    N, M = compute_NM(dataname="Cora", X=X, A=A)
    

    lb2 = []
    for i in tqdm(range(N.shape[0])):
        lb = optimize_lbd2(N[i], M[i])
        lb2.append(lb)
            
    lb2 = np.array(lb2)
    
    S = op_S(lb2, N, M)
    S = A_final(S)
    print("homophily", homophily_v2(A, gnd)) #! 0.8137
    print("homophily", homophily_v2(S, gnd)) #! 0.8691 higher than the reported result 
    print("edges of A", np.count_nonzero(A)) #! 5429
    print("edges of S", np.count_nonzero(S)) #! 9153
   
