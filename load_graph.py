import pickle
import random

import dgl
import numpy as np
import torch as th
from dgl.data import citation_graph as citegrh
from sklearn.preprocessing import MinMaxScaler
def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_labels

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g

def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = dgl.DGLGraph(data.graph)
    g.add_edges(g.nodes(), g.nodes())

    return g, features, labels, train_mask, test_mask


def load_mol_data(dsize=None, split_level=None, norm_values=False, reverse=False, undirected=False, add_self_edges=False, remove_molecule_nodes=False):
    if (dsize is None) and (split_level is None):
        print("Error.")
        exit()

    with open("test_data.pkl", 'rb') as f:
        data = pickle.load(f)

    networkx_graph = data['g']

    if remove_molecule_nodes:
        sb_nodes = list(filter(lambda x : x[1]['type'] == 'scaffold', networkx_graph.nodes(data=True)))
        networkx_graph = networkx_graph.subgraph(sb_nodes).copy()

    heirs = []
    train_mask = []
    test_mask = []
    for node in networkx_graph.nodes(data=True):
        if 'hierarchy' in node[1]:
            heirs.append(node[1]['hierarchy'])
        else:
            heirs.append(5)
        if heirs[-1] <= (5 if split_level is None else split_level):
            train_mask.append(True)
            test_mask.append(False)
        else:
            train_mask.append(False)
            test_mask.append(True)


    if add_self_edges:
        networkx_graph.add_edges_from(zip(networkx_graph.nodes(), networkx_graph.nodes())) # add self edges
    if reverse:
        networkx_graph = networkx_graph.reverse().copy()
    if undirected:
        networkx_graph = networkx_graph.to_undirected().copy()

    graph = dgl.DGLGraph(networkx_graph)

    features = th.FloatTensor(data['features'])
    if norm_values:
        labels = th.FloatTensor(MinMaxScaler().fit_transform(data['labels']))
    else:
        labels = th.FloatTensor(data['labels'])

    graph.ndata['features'] = features
    graph.ndata['labels'] = labels

    if dsize is not None:
        train_mask = [random.random() < dsize for _ in range(features.shape[0])]
        train_mask = np.array(train_mask, dtype=np.bool).flatten()
        test_mask = ~train_mask

    graph.ndata['train_mask'] = th.BoolTensor(train_mask)
    graph.ndata['test_mask'] = th.BoolTensor(test_mask)
    graph.ndata['val_mask'] = th.BoolTensor(test_mask)

    print(f"Train: {sum(train_mask)}, Test: {sum(test_mask)}")

    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    # dataloader = dgl.dataloading.NodeDataLoader(
    #     graph, np.where(train_mask)[0].flatten(), sampler,
    #     batch_size=64,
    #     shuffle=True,
    #     drop_last=False)
    #
    # test_dataloader = dgl.dataloading.NodeDataLoader(
    #     graph, np.where(test_mask)[0].flatten(), sampler,
    #     batch_size=64,
    #     shuffle=True,
    #     drop_last=False)

    return graph
