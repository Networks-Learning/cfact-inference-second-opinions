#!/usr/bin/env python

import numpy as np
import pandas as pd
from siscm import SISCM
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
    def get_proba(weights, datapoint):
        exp_wx = np.exp(np.dot(datapoint, weights))
        return exp_wx/sum(exp_wx)


    def create_data_model(self):
        #sample discrete choice model weights
        H_weights = self.rng.random( (self.n_experts, self.n_features, self.n_classes))
        #array of functions that return group probabilities
        H_marginal_proba_func = [(lambda d,w=w: SyntheticExperiment.get_proba(w, d)) for w in H_weights]

        # list of sets with group members (index)
        group_members = [ set(list(range(sum(self.group_size[:g]), sum(self.group_size[:g+1])))) for g in range(self.n_groups) ]
        self.siscm_true = SISCM("Real", self.n_classes, H_marginal_proba_func, group_members)

    def create_synthetic_data(self,N_training, N_test):
        #create dummy data
        data_train = self.rng.random((N_training, self.n_features))
        data_test = self.rng.random((N_test, self.n_features))

        #sample dummy labels
        label_train = self.siscm_true.predict(data_train, 1)
        label_test = self.siscm_true.predict(data_test, 1)

        return (data_train, data_test, label_train, label_test)

    #run synthetic experiment for values in sparsity list and #training points list
    def run_experiment(self, T, N_test, sparsity_proba_list, N_train_list):
      
      N_len = len(N_train_list)
      s_len = len(sparsity_proba_list)
      #empty matrices to store result
      scores_real = np.full((T,s_len,N_len), np.nan)
      scores_trained = np.full((T,s_len,N_len), np.nan)
      scores_naive = np.full((T,s_len,N_len), np.nan)
      scores_groups = np.full((T,s_len,N_len), np.nan)
      ratio_inedges = np.full((T,s_len,N_len), np.nan)
  
      # T experiment rounds
      for t in range(T):
        #create train and test data from true model M(Psi*)
        full_data_train, data_test, full_labels_train, label_test = self.create_synthetic_data(max(N_train_list), N_test)
        for s, sparsity_proba in enumerate(sparsity_proba_list):
            max_N = max(N_train_list)
            #sample experts' labels to drop according to sparsity probability
            missing_inds = [self.rng.choice(a=range(self.n_experts), size = ( int(sparsity_proba*self.n_experts)), replace=False) for x in range(max_N)]
            missing_inds = np.vstack(missing_inds)
            #print(missing_inds.shape)
            #leave at least two labels per datapoint
            #sample 2 expert labels to keep
            minimum_inds = [self.rng.choice(a=range(self.n_experts), size = (2), replace = False) for x in range(max_N)]
            minimum_inds = np.vstack(minimum_inds)
            #copy labels to new array
            sparse_labels = np.copy(full_labels_train)
            minimum_labels = sparse_labels[ np.arange(max_N)[:, np.newaxis], minimum_inds]
            #remove sampled experts' labels, but keep the 2 sampled labels
            sparse_labels[np.arange(max_N)[:,np.newaxis], missing_inds] = -999
            sparse_labels[np.arange(max_N)[:, np.newaxis], minimum_inds] = minimum_labels
            #sample expert's to predict during test
            test_inds = self.rng.integers(self.n_experts, size = N_test)
            #vary the number of training data
            for n, N_train in enumerate(N_train_list):
                print(t)
                print(s, " ", sparsity_proba)
                print(n, " ", N_train)

                #remove excess datapoints
                data_training = full_data_train[0:N_train]
                label_training = sparse_labels[0:N_train]

                #use true model M(Psi*) to find the ratio of edges inside the true groups
                ratio_inedges[t,s,n] = self.siscm_true.analyze_PCS_graph( data_training, label_training)
                if ratio_inedges[t,s,n]==1.0: continue
                #create and train SI-SCM M(Psi)
                siscm_psi = SISCM("SISCM_M(Psi)", self.n_classes, self.siscm_true.get_proba_function(),n_samples=500)
                siscm_psi.fit( data_training, label_training)
                
                #create SI-SCM M(H)
                siscm_H = SISCM("SISCM_M(H)", self.n_classes, self.siscm_true.get_proba_function(), siscm_H= True)

                #compare greedy algorithm groups to real groups
                scores_groups[t,s,n] = compare_groups(siscm_psi, self.siscm_true)

                #compare performace of the models for counterfactual predictions in the group
                scores_real[t,s,n] = self.siscm_true.score_counterfactuals_rand(data_test, label_test, test_inds, label_test[range(N_test),test_inds])
                scores_trained[t,s,n]= siscm_psi.score_counterfactuals_rand(data_test, label_test, test_inds, label_test[range(N_test),test_inds])
                scores_naive[t,s,n] = siscm_H.score_counterfactuals_rand(data_test, label_test, test_inds, label_test[range(N_test),test_inds])
                #print each rounds result
                print("Log Likelihood for this round:")
                print("Real: ", scores_real[t,s,n])
                print("Trained: ", scores_trained[t,s,n])
                print("Naive: ", scores_naive[t,s,n])
                print("ARI: ", scores_groups[t,s,n])
                print("Rate: ", ratio_inedges[t,s,n])
            
      #compute mean and standard deviation
      mean_score_real = np.mean(scores_real, axis=0)
      std_score_real = np.std(scores_real, axis=0)
      mean_score_trained = np.mean(scores_trained, axis=0)
      std_score_trained = np.std(scores_trained, axis=0)
      mean_score_naive = np.mean(scores_naive, axis=0)
      std_score_naive = np.std(scores_naive, axis=0)
      mean_score_groups = np.mean(scores_groups, axis=0)
      std_score_groups = np.std(scores_groups, axis=0)
      mean_rate_inedge = np.mean(ratio_inedges, axis=0)
      std_rate_inedge = np.std(ratio_inedges, axis=0)
      print()
      print("Mean Score for Group of Observed Experts")
      print("Score Naive CF: ", mean_score_naive, "+-", std_score_naive)
      print("Score CF: ", mean_score_trained, "+-", std_score_trained)
      print("Score Real CF: ", mean_score_real, "+-", std_score_real)

      #save results to file     
      df_mean_real = pd.DataFrame(mean_score_real, columns = N_train_list, index= sparsity_proba_list)
      df_mean_real.to_csv("results_synthetic/mean_real.csv")
      df_std_real = pd.DataFrame(std_score_real, columns = N_train_list, index= sparsity_proba_list)
      df_std_real.to_csv("results_synthetic/std_real.csv")
 
      df_mean_trained = pd.DataFrame(mean_score_trained, columns = N_train_list, index= sparsity_proba_list)
      df_mean_trained.to_csv("results_synthetic/mean_trained.csv")
      df_std_trained = pd.DataFrame(std_score_trained, columns = N_train_list, index= sparsity_proba_list)
      df_std_trained.to_csv("results_synthetic/std_trained.csv")
 
      df_mean_naive = pd.DataFrame(mean_score_naive, columns = N_train_list, index= sparsity_proba_list)
      df_mean_naive.to_csv("results_synthetic/mean_naive.csv")
      df_std_naive = pd.DataFrame(std_score_naive, columns = N_train_list, index= sparsity_proba_list)
      df_std_naive.to_csv("results_synthetic/std_naive.csv")
 
      df_mean_groups = pd.DataFrame(mean_score_groups, columns = N_train_list, index= sparsity_proba_list)
      df_mean_groups.to_csv("results_synthetic/mean_groups.csv")
      df_std_groups = pd.DataFrame(std_score_groups, columns = N_train_list, index= sparsity_proba_list)
      df_std_groups.to_csv("results_synthetic/std_groups.csv")
 
      df_mean_inedge = pd.DataFrame(mean_rate_inedge, columns = N_train_list, index= sparsity_proba_list)
      df_mean_inedge.to_csv("results_synthetic/mean_inedge.csv")
      df_std_inedge = pd.DataFrame(std_rate_inedge, columns = N_train_list, index= sparsity_proba_list)
      df_std_inedge.to_csv("results_synthetic/std_inedge.csv")

      #scm_model.save()
 

def main():
    seed = 44 
    #data parameters
    n_classes = 5
    n_features = 20
    #true groups parameters
    n_groups = 5
    size_max = 15   #maximum number of experts per group
    size_min = 5    #minimum number of experts per group
    #create synthetic experiment with parameters
    exp = SyntheticExperiment(n_classes, n_features, n_groups, size_max, size_min, seed)
    #number of test datapoints
    N_test=1000
    #number of experiment rounds
    T= 5
    #list of different amounts of training data
    N_train_list = [10, 20, 50, 75, 100, 150, 200, 300, 400]
    #list of different sparsity percentage/ratio
    sparsity_proba_list = [0.1, 0.3, 0.6, 0.8, 0.95]
    #run experiment for set parameter lists
    exp.run_experiment(T, N_test, sparsity_proba_list, N_train_list)

if __name__ == "__main__":
    main()
