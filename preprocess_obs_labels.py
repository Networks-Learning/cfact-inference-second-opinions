#!/usr/bin/env python

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from networkx.algorithms.approximation import max_clique

#choose number in [1, 2571] of experts to take from the data
n_nodes = 2571

df = pd.read_csv('data/cifar10_feat+labels.csv').fillna(-999)
data_all = df.filter(like='feature', axis=1).to_numpy()
labels_all = df.filter(like='chosen_label', axis=1).to_numpy(dtype = 'int')
print("Train test split")
data, data_test, labels, labels_test = train_test_split(data_all, labels_all, test_size=0.20, random_state=42)
print("Constructing edge matrix")
edges = np.ones((n_nodes,n_nodes))
for i in range(n_nodes):
    has_data = labels[labels[:,i] != -999] != -999
    edges[i,:] = np.sum(has_data, axis=0) !=0

np.fill_diagonal(edges, 0)

print("Finding the max clique")
#G = nx.Graph(edges)
#clique = max_clique(G)
clique= [1408, 2561, 2562, 2563, 2560, 2437, 1273, 1145, 646, 1929, 517, 260, 1038, 2453, 1304, 1913, 154, 926, 288, 2465, 1569, 1446, 2345, 2475, 1067, 1710, 687, 2554, 1202, 2555, 2232, 952, 825, 2491, 316, 2367, 196, 1607, 840, 1876, 2135, 1113, 2266, 224, 2275, 869, 103, 2536, 1257, 1386, 2539, 377, 1008, 2545, 2037, 2549, 757, 2553, 1530, 2299, 2173, 2559]
print(clique)
print("Saving data")
row_idx = np.any(labels[:,clique]!=-999, axis=1)
df_data = pd.DataFrame(data[row_idx])
df_data.to_csv('data/data_training.csv', index=False)
print(df_data.shape)
df_labels = pd.DataFrame(labels[row_idx][:, clique])
df_labels.to_csv('data/labels_training.csv', index=False)
row_idx_test = np.any(labels_test[:,clique]!=-999, axis=1)
df_data_test = pd.DataFrame(data_test[row_idx_test])
df_data_test.to_csv('data/data_test.csv', index=False)
print(df_data_test.shape)
df_labels_test = pd.DataFrame(labels_test[row_idx_test][:,clique])
df_labels_test.to_csv('data/labels_test.csv', index=False)

