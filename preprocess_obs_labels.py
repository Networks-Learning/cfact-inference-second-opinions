#!/usr/bin/env python

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from networkx.algorithms.approximation import max_clique

#choose number in [1, 2571] of experts to take from the data
n_nodes = 2571
#sample_size=200
#perc_disagree = 0.01

df = pd.read_csv('data/cifar10_feat+labels.csv').fillna(-999)
data_all = df.filter(like='feature', axis=1).to_numpy()
labels_all = df.filter(like='chosen_label', axis=1).to_numpy(dtype = 'int')

print("Resampling to for higher disagreement")
def has_disagreement(labels):
            val, count = np.unique(labels, return_counts=True)
            count = count[val!=-999]
            return len(count)>1

def ratio_disagreement(labels, exp):
            if labels[exp]==-999: return np.nan
            val, count = np.unique(labels, return_counts=True)
            count = count[val!=-999]
            val = val[val!=-999]
            return np.sum(count[val!=labels[exp]])/float(np.sum(count)-1)

vec_has_disagreement = np.vectorize(has_disagreement, signature='(n)->()')
sampling_mask = vec_has_disagreement(labels_all)
print("Number of resampled datapoints:", np.sum(sampling_mask))
#vec_count_disagreement = np.vectorize(lambda d: np.array([count_disagreement(d,exp) for exp in range(n_nodes)]), signature='(n)->(m)')
#count = np.nanmean(vec_count_disagreement(labels_all))
#print("disagreement ratio ", count)

data_all = data_all[sampling_mask]
labels_all = labels_all[sampling_mask]

print("Train test split")
seed = {42,3993}
data, data_test, labels, labels_test = train_test_split(data_all, labels_all, test_size=0.20, random_state=3993)

#choose random subset of experts with enough data points
has_all_labels = [np.array_equal(np.unique(labels[:,exp]),np.array([-999,0,1,2,3,4,5,6,7,8,9])) for exp in range(n_nodes)]
experts = (np.sum(labels!=-999, axis=0) > 130) & (np.sum(labels_test!=-999, axis=0) > 20) & (np.array(has_all_labels,dtype=bool))
print("# experts with data: ", np.sum(experts))
print("Experts ID:")
print(np.arange(2571,dtype=int)[experts])

labels = labels[:,experts]
labels_test = labels_test[:,experts]
row_idx = np.sum(labels!=-999, axis=1)>1
data = data[row_idx]
labels = labels[row_idx]
print(data.shape)
experts = [np.array_equal(np.unique(labels[:,exp]),np.array([-999,0,1,2,3,4,5,6,7,8,9])) for exp in range(np.sum(experts))]
print(np.sum(experts))
#experts = np.sum(labels!=-999, axis=0) > 100
#print("# experts with data: ", np.sum(experts))

labels = labels[:,experts]
labels_test = labels_test[:,experts]
#row_idx = np.sum(labels!=-999, axis=1)>1
#data = data[row_idx]
#labels = labels[row_idx]
#experts = [np.array_equal(np.unique(labels[:,exp]),np.array([-999,0,1,2,3,4,5,6,7,8,9])) for exp in range(np.sum(experts))]
print(np.sum(experts))
print(np.sum(np.sum(labels!=-999, axis=1)==1))
print(np.sum(np.sum(labels!=-999, axis=1)<1))
#labels_test = labels_test[:,experts]
row_idx_test = np.sum(labels_test!=-999, axis=1)>1
data_test = data_test[row_idx_test]
labels_test = labels_test[row_idx_test]

has_disagreement = vec_has_disagreement(labels)
print("Number of full agreement:", data.shape[0]-np.sum(has_disagreement))
has_disagreement = vec_has_disagreement(labels_test)
print("Number of full agreement:", data_test.shape[0]-np.sum(has_disagreement))

vec_ratio_disagreement = np.vectorize(lambda d: np.array([ratio_disagreement(d,exp) for exp in range(labels.shape[1])]), signature='(n)->(m)')
ratio = np.nanmean(vec_ratio_disagreement(labels))
print("disagreement ratio ", ratio)
ratio = np.nanmean(vec_ratio_disagreement(labels_test))
print("disagreement ratio ", ratio)


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

