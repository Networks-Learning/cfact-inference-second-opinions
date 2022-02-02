#!/usr/bin/env python

import numpy as np
import pandas as pd

#choose number in [1, 2571] of experts to take from the data
n_experts = 2571

labels = pd.read_csv('data/cifar10h-raw.csv', usecols= ['annotator_id','is_attn_check','chosen_label','cifar10_test_test_idx'])
#take out attention checks
labels = labels[labels.is_attn_check==0]
labels = labels[labels.annotator_id < n_experts]
print(labels)

npz = np.load('features/CIFAR10_vgg19-keras_features.npz')
#df_features = pd.DataFrame(features)
df_features = pd.DataFrame.from_dict({idx: item for idx, item in enumerate(npz['features_testing'])}, orient='index' )

df_all = df_features.add_prefix('feature_')
print(df_all)
for n in range(n_experts):
    df_annotator = labels[labels.annotator_id == n].set_index('cifar10_test_test_idx')
    df_all = df_all.join(df_annotator['chosen_label'], how = 'left' , rsuffix="_" + str(n))

df_all.rename(columns={'chosen_label': 'chosen_label_0'})
df_all.to_csv('data/cifar10_feat+labels.csv', index=False)

#extra for analysing the label annotations
df_all=df_all.drop(df_all.filter(regex='feature_').columns, axis=1)
print(df_all)
#print(df_all.dropna(thresh=3))
df_nonnan = df_all.count(axis=1)
print(df_nonnan[df_nonnan>0])
print(df_all.count(axis=1).mean())
df_unique_labels=df_all.stack().groupby(level=0).apply(lambda x: len(x.unique().tolist()))
print(df_unique_labels[df_unique_labels>1])
print(df_unique_labels[df_unique_labels>1].mean())

