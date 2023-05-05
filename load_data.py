import torch
import numpy as np
from torch_geometric.datasets import WebKB, AttributedGraphDataset, WikipediaNetwork, Amazon
from scipy.sparse import coo_matrix
import scipy.io as sio
import scipy.sparse as sp
import networkx as nx
import torch_geometric.transforms as T

def WebKB_graph(dataname='Texas') :
    dataset = WebKB(root='./data/webKB/{}'.format(dataname), name="{}".format(dataname))
    data = dataset[0]
    X = data.x.numpy()
    gnd = data.y.numpy()
    N = X.shape[0]
    row = data.edge_index[0].numpy()
    col = data.edge_index[1].numpy()
    M = data.num_edges
    values = torch.ones(M)
    adj = coo_matrix((values, (row, col)), shape=(N, N))
    adj = adj.todense()
    adj = np.array(adj)
    return X, adj, gnd

def Amazon_graph(dataname=''):
    dataset = Amazon(root='./data/amazon/{}'.format(dataname), name="{}".format(dataname))
    data = dataset[0]
    X = data.x.numpy()
    gnd = data.y.numpy()
    N = X.shape[0]
    # train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    # print(N)
    # A = np.zeros((N, N))
    row = data.edge_index[0].numpy()
    col = data.edge_index[1].numpy()
    M = data.num_edges
    values = torch.ones(M)
    adj = coo_matrix((values, (row, col)), shape=(N, N))
    adj = adj.todense()
    adj = np.array(adj)
    # print(np.count_nonzero(adj))
    return X, adj, gnd

def Attributed_graph(dataname='Cora'):
    print(dataname)
    dataset = AttributedGraphDataset(root='./data/attr/{}'.format(dataname), name="{}".format(dataname))
    data = dataset[0]
    X = data.x.numpy()
    gnd = data.y.numpy()
    N = X.shape[0]
    # train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    # print(N)
    # A = np.zeros((N, N))
    row = data.edge_index[0].numpy()
    col = data.edge_index[1].numpy()
    M = data.num_edges
    values = torch.ones(M)
    adj = coo_matrix((values, (row, col)), shape=(N, N))
    adj = adj.todense()
    adj = np.array(adj)
    # print(np.count_nonzero(adj))

    return X, adj, gnd

def Wiki_graph(dataname='Squirrel'):
    dataset = WikipediaNetwork(root='./data/wiki/{}'.format(dataname), name="{}".format(dataname))
    data = dataset[0]
    X = data.x.numpy()
    gnd = data.y.numpy()
    N = X.shape[0]
    # print(N)
    # train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    # A = np.zeros((N, N))
    row = data.edge_index[0].numpy()
    col = data.edge_index[1].numpy()
    M = data.num_edges
    values = torch.ones(M)
    adj = coo_matrix((values, (row, col)), shape=(N, N))
    adj = adj.todense()
    adj = np.array(adj)
    # print(np.count_nonzero(adj))

    return X, adj, gnd

def Acm(dataname='ACM') :
    if dataname == "ACM" :
        # Load data
        dataset = "./Data/mat/" + 'ACM3025'
        data = sio.loadmat('{}.mat'.format(dataset))
        if (dataset == 'large_cora') :
            X = data['X']
            A = data['G']
            gnd = data['labels']
            gnd = gnd[0, :]
        else :
            X = data['feature']
            A = data['PAP']
            B = data['PLP']
            # C = data['PMP']
            # D = data['PTP']
    if sp.issparse(X) :
        X = X.todense()
    # X_ = []
    X = np.array(X)
    A = np.array(A)
   

    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)

    return X, A, gnd

def MultiView_Attributed_graph(dataname='ACM'):
    if dataname == "ACM":
        X, A, gnd = Acm()
    else:
        X, A, gnd = None, None, None
    return X, A, gnd

def Airports_networks(dataset_str, data_path='./data/Airports/') :
    """Read the data and preprocess the task information."""
    dataset_G = data_path + "{}-airports.edgelist".format(dataset_str)
    dataset_L = data_path + "labels-{}-airports.txt".format(dataset_str)
    label_raw, nodes = [], []
    with open(dataset_L, 'r') as file_to_read :
        while True :
            lines = file_to_read.readline()
            if not lines :
                break
            node, label = lines.split()
            if label == 'label' :
                continue
            label_raw.append(int(label))
            nodes.append(int(node))
    label_raw = np.array(label_raw)
    # print(label_raw)
    G = nx.read_edgelist(open(dataset_G, 'rb'), nodetype=int)
    adj = nx.adjacency_matrix(G, nodelist=nodes)

    # task information
    degreeNode = np.sum(adj, axis=1).A1
    degreeNode = degreeNode.astype(np.int32)
    features = np.zeros((degreeNode.size, degreeNode.max() + 1))
    features[np.arange(degreeNode.size), degreeNode] = 1
    features = sp.csr_matrix(features)
    features = features.A
    adj = np.array(adj.todense())
    print(features.shape, adj.shape)
    # label_raw = np

    return features, adj, label_raw


Swither = {
        0: WebKB_graph,
        1: Amazon_graph,
        2: Attributed_graph,
        3: Wiki_graph,
        5: MultiView_Attributed_graph,
        6: Airports_networks
        }

if __name__ == '__main__':
    X, adj, gnd = Swither[2]("Wiki")
    print(X.shape, np.count_nonzero(adj), len(np.unique(gnd)))
    X, adj, gnd = Swither[2]("Cora")
    print(X.shape, np.count_nonzero(adj))