#!/usr/bin/env python

import numpy as np
import sklearn

rng = np.random.default_rng(42)
n_classes = 5

#retrieves categorical probabilities for individuals
def log_prob(weights, datapoint):
    exp_wx = np.exp(np.dot(datapoint, weights))
    return np.log(exp_wx/sum(exp_wx))

#retrieves categorical probabilities for groups
def group_log_prob( group_weights, datapoint):
    return np.array(list(map((lambda w: log_prob(w, datapoint)), group_weights)))

#samples predictions per group using the Gumbel Trick
def sample_group_predictions(group_log_prob):
    mu, beta = 0, 1
    u = np.random.gumbel(mu, beta, n_classes)
    return np.apply_along_axis( (lambda log_p: np.argmax( log_p + u)), 1, group_log_prob)


#create dummy data
N_training=10
N_test=5
n_features = 6
data_training = rng.random((N_training, n_features))
data_test = rng.random((N_test, n_features))

#create synthetic experts
n_subgroups = 10
subgroup_size = rng.integers(3,10,size=n_subgroups)
H_total = sum(subgroup_size)
print("size of expert subgroups: ", subgroup_size)

H_weights = [ rng.random( (s, n_features, n_classes)) for s in subgroup_size ]

PCS_graph = np.ones((H_total,H_total))
for ind in range(3):
    H_log_prob = [group_log_prob(w, data_training[ind,:]) for w in H_weights]
    H_predictions = [ sample_group_predictions(group_log_prob) for group_log_prob in H_log_prob]
    print(H_predictions)
    
    #contruct matrices of the whole expert groups for probabilities and predictions
    prob_matrix = np.exp(np.concatenate(H_log_prob, 0))
    prediction_vector = np.concatenate(H_predictions, 0)
    #function for checking PCS?
    for i in range(H_total):
        for j in range(H_total):
            #check that PCS condition is satisfied
            ci= prediction_vector[i]
            cj= prediction_vector[j]
            r_ci = prob_matrix[j,ci]/ prob_matrix[i,ci]
            r_cj = prob_matrix[j,cj]/ prob_matrix[i,cj]
            if r_ci >= r_cj and ci!=cj:
                #otherwise remove edge
                PCS_graph[i,j]=0


#greedy algorithm to find clique


