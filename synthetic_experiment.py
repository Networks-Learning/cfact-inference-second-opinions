#!/usr/bin/env python

import numpy as np
import pandas as pd
from scm import SCM
from helper import *

class SyntheticExperiment:
    def __init__(self, n_classes, n_features, n_groups, size_max, size_min, seed=44):
        self.rng = np.random.default_rng(seed)
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_groups = n_groups

        #create synthetic experts
        self.group_size = self.rng.integers(size_min,size_max,size=self.n_groups)
        self.n_experts = sum(self.group_size)
        print("size of expert groups: ", self.group_size)
        print("Total number of experts: ", self.n_experts)
        self.create_data_model()


    #retrieves categorical probabilities for individuals
    def get_prob(weights, datapoint):
        exp_wx = np.exp(np.dot(datapoint, weights))
        return exp_wx/sum(exp_wx)


    def create_data_model(self):
        #sample discrete choice model weights
        H_weights = self.rng.random( (self.n_experts, self.n_features, self.n_classes))
        #array of functions that return group probabilities
        H_prob_functions = [(lambda d,w=w: SyntheticExperiment.get_prob(w, d)) for w in H_weights]

        # list of sets with group members (index)
        group_members = [ set(list(range(sum(self.group_size[:g]), sum(self.group_size[:g+1])))) for g in range(self.n_groups) ]
        self.scm_real_model = SCM("Real", self.n_classes, H_prob_functions, group_members)

    def create_synthetic_data(self,N_training, N_test):
        #create dummy data
        data_training = self.rng.random((N_training, self.n_features))
        data_test = self.rng.random((N_test, self.n_features))

        #sample dummy labels
        label_training = self.scm_real_model.predict(data_training, 1)
        label_test = self.scm_real_model.predict(data_test, 1)

        return (data_training, data_test, label_training, label_test)

    def run_experiment(self, T, N_test, sparsity_prob_list, N_training_list):
      
      N_len = len(N_training_list)
      s_len = len(sparsity_prob_list)
      scores_real = np.full((T,s_len,N_len), np.nan)
      scores_trained = np.full((T,s_len,N_len), np.nan)
      scores_naive = np.full((T,s_len,N_len), np.nan)
      scores_groups = np.full((T,s_len,N_len), np.nan)
      rate_inedges = np.full((T,s_len,N_len), np.nan)
  
      for t in range(T):
        full_data_training, data_test, full_label_training, label_test = self.create_synthetic_data(max(N_training_list), N_test)
        for s, sparsity_prob in enumerate(sparsity_prob_list):
            max_N = max(N_training_list)
            missing_inds = [self.rng.choice(a=range(self.n_experts), size = ( int(sparsity_prob*self.n_experts)), replace=False) for x in range(max_N)]
            missing_inds = np.vstack(missing_inds)
            #print(missing_inds.shape)
            minimum_inds = [self.rng.choice(a=range(self.n_experts), size = (2), replace = False) for x in range(max_N)]
            minimum_inds = np.vstack(minimum_inds)
            sparse_labels = np.copy(full_label_training)
            minimum_labels = sparse_labels[ np.arange(max_N)[:, np.newaxis], minimum_inds]
            sparse_labels[np.arange(max_N)[:,np.newaxis], missing_inds] = -999
            #print(missing_inds)
            #print(sparse_labels)
            #print(sparse_labels[:, missing_inds])
            sparse_labels[np.arange(max_N)[:, np.newaxis], minimum_inds] = minimum_labels
            test_inds = self.rng.integers(self.n_experts, size = N_test)
            for n, N_training in enumerate(N_training_list):
                print(t)
                print(s, " ", sparsity_prob)
                print(n, " ", N_training)

                data_training = full_data_training[0:N_training]
                label_training = sparse_labels[0:N_training]

                rate_inedges[t,s,n] = self.scm_real_model.analyze_PCS_graph( data_training, label_training)
                if rate_inedges[t,s,n]==1.0: continue
                #trained SCM
                scm_model = SCM("Trained", self.n_classes, self.scm_real_model.get_prob_function(),n_samples=500)
                scm_model.fit( data_training, label_training)
                
                #naive SCM
                scm_naive = SCM("Naive", self.n_classes, self.scm_real_model.get_prob_function(), naive= True)

                #compare greedy algorithm groups to real groups
                scores_groups[t,s,n] = compare_groups(scm_model, self.scm_real_model)

                #compare performace of the models for counterfactual predictions in the group
                # score of counterfactual labels for the real group of the observed expert and non cf. labels for remaining experts in the trained scm
                # group of the observed expert 
                scores_real[t,s,n] = self.scm_real_model.score_counterfactuals_rand(data_test, label_test, test_inds, label_test[range(N_test),test_inds])
                scores_trained[t,s,n]= scm_model.score_counterfactuals_rand(data_test, label_test, test_inds, label_test[range(N_test),test_inds])
                scores_naive[t,s,n] = scm_naive.score_counterfactuals_rand(data_test, label_test, test_inds, label_test[range(N_test),test_inds])
                print("Log Likelihood for this round:")
                print("Real: ", scores_real[t,s,n])
                print("Trained: ", scores_trained[t,s,n])
                print("Naive: ", scores_naive[t,s,n])
                print("ARI: ", scores_groups[t,s,n])
                print("Rate: ", rate_inedges[t,s,n])
            

      mean_score_real = np.mean(scores_real, axis=0)
      std_score_real = np.std(scores_real, axis=0)
      mean_score_trained = np.mean(scores_trained, axis=0)
      std_score_trained = np.std(scores_trained, axis=0)
      mean_score_naive = np.mean(scores_naive, axis=0)
      std_score_naive = np.std(scores_naive, axis=0)
      mean_score_groups = np.mean(scores_groups, axis=0)
      std_score_groups = np.std(scores_groups, axis=0)
      mean_rate_inedge = np.mean(rate_inedges, axis=0)
      std_rate_inedge = np.std(rate_inedges, axis=0)
      print()
      print("Mean Score for Group of Observed Experts")
      print("Score Naive CF: ", mean_score_naive, "+-", std_score_naive)
      print("Score CF: ", mean_score_trained, "+-", std_score_trained)
      print("Score Real CF: ", mean_score_real, "+-", std_score_real)

     
      df_mean_real = pd.DataFrame(mean_score_real, columns = N_training_list, index= sparsity_prob_list)
      df_mean_real.to_csv("results_synthetic/mean_real.csv")
      df_std_real = pd.DataFrame(std_score_real, columns = N_training_list, index= sparsity_prob_list)
      df_std_real.to_csv("results_synthetic/std_real.csv")
 
      df_mean_trained = pd.DataFrame(mean_score_trained, columns = N_training_list, index= sparsity_prob_list)
      df_mean_trained.to_csv("results_synthetic/mean_trained.csv")
      df_std_trained = pd.DataFrame(std_score_trained, columns = N_training_list, index= sparsity_prob_list)
      df_std_trained.to_csv("results_synthetic/std_trained.csv")
 
      df_mean_naive = pd.DataFrame(mean_score_naive, columns = N_training_list, index= sparsity_prob_list)
      df_mean_naive.to_csv("results_synthetic/mean_naive.csv")
      df_std_naive = pd.DataFrame(std_score_naive, columns = N_training_list, index= sparsity_prob_list)
      df_std_naive.to_csv("results_synthetic/std_naive.csv")
 
      df_mean_groups = pd.DataFrame(mean_score_groups, columns = N_training_list, index= sparsity_prob_list)
      df_mean_groups.to_csv("results_synthetic/mean_groups.csv")
      df_std_groups = pd.DataFrame(std_score_groups, columns = N_training_list, index= sparsity_prob_list)
      df_std_groups.to_csv("results_synthetic/std_groups.csv")
 
      df_mean_inedge = pd.DataFrame(mean_rate_inedge, columns = N_training_list, index= sparsity_prob_list)
      df_mean_inedge.to_csv("results_synthetic/mean_inedge.csv")
      df_std_inedge = pd.DataFrame(std_rate_inedge, columns = N_training_list, index= sparsity_prob_list)
      df_std_inedge.to_csv("results_synthetic/std_inedge.csv")

      scm_model.save()
 

def main():
    n_classes = 5
    n_features = 20
    n_groups = 5
    size_max = 15
    size_min = 5
    seed = 44 
    exp = SyntheticExperiment(n_classes, n_features, n_groups, size_max, size_min, seed)
    N_test=100
    T= 5
    N_training_list = [10, 20, 50, 75, 100, 150, 200, 300, 400]
    sparsity_prob_list = [0.1, 0.3, 0.6, 0.8, 0.95]
    exp.run_experiment(T, N_test, sparsity_prob_list, N_training_list)

if __name__ == "__main__":
    main()
