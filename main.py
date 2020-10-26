DEVICE = 'cpu'

import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import metrics
import random
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.nn
import torch
from dgl.data import citation_graph as citegrh
import networkx as nx

import time
import numpy as np
from tqdm import tqdm

from load_graph import load_mol_data

def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super(GCN, self).__init__()
        self.d_layer1 = nn.Linear(in_dim, hidden_dim)
        self.n1 = nn.BatchNorm1d(hidden_dim)
        self.g_layer1 = dgl.nn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean', activation=F.relu)
        self.d_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.n2 = nn.BatchNorm1d(hidden_dim)
        self.g_layer2 = dgl.nn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean', activation=F.relu)
        self.d_layer3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)



    def forward(self, g, h):
        h = self.d_layer1(h)
        h = self.n1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.g_layer1(g,h)

        h = self.d_layer2(h)
        h = self.n2(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.g_layer2(g, h)

        h = F.relu(self.d_layer3(h))

        return h


def evaluate(model, g, features, labels, mask, G):
    model.eval()
    with th.no_grad():
        logp = net(g, features)
        preds, labels = logp[mask], labels[mask]
        print(preds.shape, labels.shape)
        loss = F.mse_loss(preds, labels)

        preds, labels = preds.detach().cpu().numpy(), labels.detach().cpu().numpy()

        print('r2', metrics.r2_score(labels, preds))


        # nx.draw(G, pos=nx.circular_layout(G), node_color=labels_)
        # plt.show()
        #
        # labels = labels[mask]
        # logp = logp[mask]



        return loss




g = load_mol_data(0.8)

net = GCN(1302, 32, 1).to(DEVICE)

print("trainbale:", trainable_count(net))
optimizer = th.optim.Adam(net.parameters(), lr=5e-4)
dur = []
t0 = time.time()

tmask = g.ndata['train_mask']
g = g.to(DEVICE)
g.ndata['features'] = g.ndata['features'].to(DEVICE)
g.ndata['labels'] = g.ndata['labels'].to(DEVICE)
g.ndata['train_mask'] = g.ndata['train_mask'].to(DEVICE)
g.ndata['test_mask'] = g.ndata['test_mask'].to(DEVICE)

for epoch in range(10000):
    net.train()

    for i in tqdm(range(200)):

        if epoch >= 3:
            dur.append(time.time() - t0)


        if epoch >=3:
            t0 = time.time()
        optimizer.zero_grad()

        logits = net(g, g.ndata['features'])
        logp = (logits)

        rid = th.LongTensor(np.random.randint(0, len(np.where(tmask)[0]), 128)).to(DEVICE)

        loss = F.mse_loss(logp[g.ndata['train_mask']][rid], g.ndata['labels'][g.ndata['train_mask']][rid])

        loss.backward()
        optimizer.step()

    optimizer.zero_grad()

    logits = net(g, g.ndata['features'])
    logp = (logits)


    loss = F.mse_loss(logp[g.ndata['train_mask']], g.ndata['labels'][g.ndata['train_mask']])

    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)

    acc = evaluate(net, g, g.ndata['features'], g.ndata['labels'], g.ndata['test_mask'], g)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))
