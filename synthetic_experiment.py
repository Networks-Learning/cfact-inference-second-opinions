#!/usr/bin/env python

import numpy as np
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
      scores_real = np.zeros((T,s_len,N_len))
      scores_trained = np.zeros((T,s_len,N_len))
      scores_naive = np.zeros((T,s_len,N_len))
      scores_groups = np.zeros((T,s_len,N_len))
      rate_inedges = np.zeros((T,s_len,N_len))
  
      for t in range(T):
        full_data_training, data_test, full_label_training, label_test = self.create_synthetic_data(max(N_training_list), N_test)
        #fig, axs = plt.subplots(s_len, N_len)
        for s, sparsity_prob in enumerate(sparsity_prob_list):
            max_N = max(N_training_list)
            missing_inds = [self.rng.choice(a=range(self.n_experts), size = ( int(sparsity_prob*self.n_experts)), replace=False) for x in range(max_N)]
            missing_inds = np.vstack(missing_inds)
            print(missing_inds.shape)
            minimum_inds = [self.rng.choice(a=range(self.n_experts), size = (2), replace = False) for x in range(max_N)]
            minimum_inds = np.vstack(minimum_inds)
            sparse_labels = np.copy(full_label_training)
            minimum_labels = sparse_labels[ np.arange(max_N)[:, np.newaxis], minimum_inds]
            sparse_labels[np.arange(max_N)[:,np.newaxis], missing_inds] = -999
            #print(missing_inds)
            print(sparse_labels)
            #print(sparse_labels[:, missing_inds])
            sparse_labels[np.arange(max_N)[:, np.newaxis], minimum_inds] = minimum_labels
            test_inds = self.rng.integers(self.n_experts, size = N_test)
            for n, N_training in enumerate(N_training_list):
                print(t)
                print(s, " ", sparsity_prob)
                print(n, " ", N_training)

                data_training = full_data_training[0:N_training]
                label_training = sparse_labels[0:N_training]

                #trained SCM
                scm_model = SCM("Trained", self.n_classes, self.scm_real_model.get_prob_function())
                scm_model.fit( data_training, label_training)
                
                rate_inedges[t,s,n] = self.scm_real_model.analyze_PCS_graph( data_training, label_training)
                #naive SCM
                scm_naive = SCM("Naive", self.n_classes, self.scm_real_model.get_prob_function(), naive= True)

                #compare greedy algorithm groups to real groups
                scores_groups[t,s,n] = compare_groups(scm_model, self.scm_real_model)

                #compare performace of the models for counterfactual predictions in the group
                #groups = [scm_model.get_group(ind) for ind in obs_inds]
                #groups_real = [self.scm_real_model.get_group(ind) for ind in obs_inds]
                # score of counterfactual labels for the real group of the observed expert and non cf. labels for remaining experts in the trained scm
                # group of the observed expert 
                scores_real[t,s,n] = self.scm_real_model.score_counterfactuals_rand(data_test, label_test, test_inds, label_test[range(N_test),test_inds])
                scores_trained[t,s,n]= scm_model.score_counterfactuals_rand(data_test, label_test, test_inds, label_test[range(N_test),test_inds])
                scores_naive[t,s,n] = scm_naive.score_counterfactuals_rand(data_test, label_test, test_inds, label_test[range(N_test),test_inds])

                """
                obs_inds = self.rng.integers(self.n_experts, size = N_test)
                scores_trained[t,s,n]= scm_model.score_counterfactuals(data_test, label_test, obs_inds, label_test[range(N_test),obs_inds])
                scores_real[t,s,n]= self.scm_real_model.score_counterfactuals(data_test, label_test, obs_inds, label_test[range(N_test),obs_inds])
                scores_naive[t,s,n]= scm_naive.score_counterfactuals(data_test, label_test, obs_inds, label_test[range(N_test),obs_inds])
                scores_trained[t,s,n]= scm_model.score(data_test, label_test)
                scores_real[t,s,n]= self.scm_real_model.score(data_test, label_test)
                scores_naive[t,s,n]= scm_naive.score(data_test, label_test)
                """
                #scm_model.print_number_failed_attemps()
            

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

      plot(sparsity_prob_list, N_training_list, mean_score_real, mean_score_trained, mean_score_naive, std_score_real, std_score_trained, std_score_naive)
      plot_group_comparison(sparsity_prob_list, N_training_list, mean_score_groups, std_score_groups)
      plot_rate_inedge(sparsity_prob_list, N_training_list, mean_rate_inedge, std_rate_inedge)
      scm_model.save()


def main():
    n_classes = 5
    n_features = 20
    n_groups = 6
    size_max = 15
    size_min = 5
    seed = 44 
    exp = SyntheticExperiment(n_classes, n_features, n_groups, size_max, size_min, seed)
    N_test=1000
    T= 5
    N_training_list = [5, 20, 70, 100, 200, 400]#[5,20,60,90,120] 
    sparsity_prob_list = [0.0, 0.3, 0.6, 0.8, 0.95]#list(np.linspace(0, 1, num=5, endpoint=False))
    exp.run_experiment(T, N_test, sparsity_prob_list, N_training_list)

if __name__ == "__main__":
    main()
