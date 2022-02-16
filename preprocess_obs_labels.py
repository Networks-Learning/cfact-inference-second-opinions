#!/usr/bin/env python

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from networkx.algorithms.approximation import max_clique

#choose number in [1, 2571] of experts to take from the data
n_nodes = 2571
sample_size=200
perc_disagree = 0.05

df = pd.read_csv('data/cifar10_feat+labels.csv').fillna(-999)
data_all = df.filter(like='feature', axis=1).to_numpy()
labels_all = df.filter(like='chosen_label', axis=1).to_numpy(dtype = 'int')
#choose random subset of experts
rng = np.random.default_rng(42)
experts = rng.choice(n_nodes, sample_size, replace=False)
labels_all = labels_all[:,experts]
row_idx = np.any(labels_all!=-999, axis=1)
data_all = data_all[row_idx]
labels_all = labels_all[row_idx]
print("Resampling to ", perc_disagree, "% disagreement")
def has_disagreement(labels, perc_disagree=perc_disagree):
            val, count = np.unique(labels, return_counts=True)
            count = count[val!=-999]
            agreement_ratio = np.max(count)/float(np.sum(count))
            return agreement_ratio < (1-perc_disagree)
vec_has_disagreement = np.vectorize(has_disagreement, signature='(n)->()')
sampling_mask = vec_has_disagreement(labels_all)
print("Number of resampled datapoints:", np.sum(sampling_mask))
data_all = data_all[sampling_mask]
labels_all = labels_all[sampling_mask]
print("Train test split")
data, data_test, labels, labels_test = train_test_split(data_all, labels_all, test_size=0.20, random_state=42)
"""
print("Constructing edge matrix")
edges = np.ones((n_nodes,n_nodes))
for i in range(n_nodes):
    has_data = labels[labels[:,i] != -999] != -999
    edges[i,:] = np.sum(has_data, axis=0) !=0

np.fill_diagonal(edges, 0)

print("Finding the max clique")
#G = nx.Graph(edges)
#clique = list(max_clique(G))
#clique all data
#clique= [1408, 2561, 2562, 2563, 2560, 2437, 1273, 1145, 646, 1929, 517, 260, 1038, 2453, 1304, 1913, 154, 926, 288, 2465, 1569, 1446, 2345, 2475, 1067, 1710, 687, 2554, 1202, 2555, 2232, 952, 825, 2491, 316, 2367, 196, 1607, 840, 1876, 2135, 1113, 2266, 224, 2275, 869, 103, 2536, 1257, 1386, 2539, 377, 1008, 2545, 2037, 2549, 757, 2553, 1530, 2299, 2173, 2559]
#clique 0.15% disagree
#clique =[ 2048, 1665, 1, 2179, 521, 2443, 781, 1295, 1296, 17, 1299, 1172, 1813, 2325, 1692, 668, 286, 416, 1953, 2081, 1062, 1705, 2475, 306, 1977, 441, 2108, 193, 1858, 2242, 1224, 1353, 2378, 975, 1103, 1361, 2006, 88, 1754, 987, 735, 1249, 866, 1510, 632, 1134, 1521, 114, 885, 248, 1914, 1023]
#clique 0.05% disagree
clique=[2048, 1665, 1, 1037, 911, 1296, 1295, 1167, 1299, 17, 1813, 920, 665, 1692, 1914, 286, 416, 1953, 1056, 2081, 544, 550, 1705, 306, 950, 441, 1977, 2108, 193, 1858, 2371, 1353, 2506, 1099, 2382, 1361, 725, 2006, 600, 88, 1754, 2145, 1250, 1510, 1006, 2286, 1135, 1521, 626, 114, 248, 2426]
print(clique)
print("Saving data")
row_idx = np.any(labels[:,clique]!=-999, axis=1)
print(row_idx.shape)
df_data = pd.DataFrame(data[row_idx])
df_data.to_csv('data/data_training.csv', index=False)
print(df_data.shape)
df_labels = pd.DataFrame(labels[row_idx][:, clique])
df_labels.to_csv('data/labels_training.csv', index=False)
print(np.sum(labels[row_idx][:,clique]!=-999, axis=0))
print(np.sum(labels[row_idx][:,clique]!=-999, axis=1))
row_idx_test = np.any(labels_test[:,clique]!=-999, axis=1)
df_data_test = pd.DataFrame(data_test[row_idx_test])
df_data_test.to_csv('data/data_test.csv', index=False)
print(df_data_test.shape)
df_labels_test = pd.DataFrame(labels_test[row_idx_test][:,clique])
df_labels_test.to_csv('data/labels_test.csv', index=False)
print(np.sum(labels_test[row_idx_test][:,clique]!=-999, axis=1))
"""
print("Saving data")
df_data = pd.DataFrame(data)
df_data.to_csv('data/data_training.csv', index=False)
print(df_data.shape)
df_labels = pd.DataFrame(labels)
df_labels.to_csv('data/labels_training.csv', index=False)
print(np.sum(labels!=-999, axis=0))
print(np.sum(labels!=-999, axis=1))
df_data_test = pd.DataFrame(data_test)
df_data_test.to_csv('data/data_test.csv', index=False)
print(df_data_test.shape)
df_labels_test = pd.DataFrame(labels_test)
df_labels_test.to_csv('data/labels_test.csv', index=False)
print(np.sum(labels_test!=-999, axis=1))
