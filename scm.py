#!/usr/bin/env python

import numpy as np
import random
from PCS_graph import PCS_graph
import math 
from sklearn.metrics import top_k_accuracy_score

import warnings
from functools import partial
from time import perf_counter


class SCM:
    list_prob_functions = []

    def __init__(self, name, n_classes,  list_prob_functions, group_members = [], naive = False, n_samples =1):
        self.name = name
        self.n_classes = n_classes
        self.n_samples = n_samples
        #experts' marginal probability functions
        self.list_prob_functions = list_prob_functions
        self.n_experts = len(list_prob_functions)
        #Gumbel Max parameters
        self.mu, self.beta = 0, 1
        self.rng = np.random.default_rng(42)
        # PCS graph for training
        self.graph = None

        #optional when groups are know (e.g., sampling synthetic data)
        if naive:
            group_members = [ set([ind]) for ind in range(self.n_experts)] 
        self.set_group_membership(group_members)

    #helper functions
    def set_group_membership(self, group_membership_list):
        self.group_members = group_membership_list
        self.group_members_sorted = [ sorted(group) for group in group_membership_list]
        print("Partition of Model ", self.name)
        print(self.group_members_sorted)
        self.n_groups= len(group_membership_list)

        # construct index dictionary from group membership list
        index_dict = {}
        for g in range(len(self.group_members_sorted)):
            i = 0
            for index in self.group_members_sorted[g]:
                index_dict[index] = g, i
                i+=1
        self.index_dict = index_dict

    #returns group member set this individual belongs to
    def get_group(self, ind):
        return self.group_members[self.index_dict[ind][0]]

    def get_group_index(self, ind):
        return self.index_dict[ind][0]

    def get_group_membership_list(self):
        return self.group_members

    #return experts probabilitity functions
    def get_prob_function(self):
        return self.list_prob_functions

    #compute individual experts probabilities for datapoint
    def get_prob(self, datapoint):
        #return [ np.asarray([np.log(f(datapoint)) for f in self.list_prob_functions])]
        return [ np.asarray([f(datapoint) for f in self.list_prob_functions])]

    #compute experts probabilities for datapoint, returns it as group matrix
    #note group_membership need to be list of lists not list of sets otherwise iteration order is not guaranteed
    def get_group_prob(self, datapoint, group_membership):
        group_prob = []
        for g in range(len(group_membership)):
            list_prob = [ self.list_prob_functions[ind](datapoint) for ind in group_membership[g]]
            group_prob.append( np.asarray( list_prob ))
        return group_prob

    #sample group prediction from prior
    def sample_group_predictions(self, group_prob, times=..., return_prob=False):
        if times is ...: times = self.n_samples
        
        def count(pred_ind):
            val, count = np.unique(pred_ind, return_counts=True)
            full_count = np.zeros((self.n_classes))
            full_count[val] = count
            return full_count

        u = self.rng.gumbel(self.mu, self.beta, size=(times, self.n_classes))
        argmax = np.vectorize((lambda p: np.argmax( np.log(p) + u, axis=1)), signature='(n)->(t)')
        prediction = argmax( group_prob)
        count_vec = np.vectorize(count, signature='(n)->(k)')
        counter = count_vec(prediction)
       
        if return_prob:
            return counter.astype("float")/float(times)
        
        return np.argmax(counter, axis=1)

    
    def sample_gumbels(self,trans_probabilities, s_p_real, num_of_samples):
        
        #############################################
        # This part is adapted from https://cmaddis.github.io/gumbel-machinery
        
        def truncated_gumbel(alpha, truncation, rng):
            gumbel = rng.gumbel() + np.log(alpha)
            return -np.log(np.exp(-gumbel) + np.exp(-truncation))
        
        def topdown(alphas, k, rng):
            topgumbel = rng.gumbel() + np.log(sum(alphas))
            gumbels = []
            for i in range(len(alphas)):
                if i == k:
                    gumbel = topgumbel - np.log(trans_probabilities[i])
                elif trans_probabilities[i]!=0:
                    gumbel = truncated_gumbel(alphas[i], topgumbel, rng) - np.log(trans_probabilities[i])
                else:
                    gumbel = rng.gumbel() # When the probability is zero, sample an unconstrained Gumbel

                gumbels.append(gumbel)
            return gumbels
        #############################################
        #gumbels = [topdown(trans_probabilities, s_p_real, np.random.default_rng(seed+1)) for seed in range(num_of_samples)]
        gumbels = [topdown(trans_probabilities, s_p_real, self.rng) for seed in range(num_of_samples)]

        # Sanity check
        for gum in gumbels:
            temp = gum + np.log(trans_probabilities)
            assert np.argmax(temp)==s_p_real, "Sampled gumbels don't match with realized argmax"
        
        return gumbels
    
    #sample group prediction from posterior
    #note g_ind is the observed individuals index in the group not overall
    def sample_counterfactual_predictions(self, group_prob, g_ind, label, times=..., return_prob=False):
        if times is ...: times = self.n_samples
        
        def count(pred_ind):
            val, count = np.unique(pred_ind, return_counts=True)
            full_count = np.zeros((self.n_classes))
            full_count[val] = count
            return full_count

        u = self.sample_gumbels(group_prob[g_ind,:], label, times)
        argmax = np.vectorize((lambda p: np.argmax( np.log(p) + u, axis=1)), signature='(n)->(t)')
        prediction = argmax( group_prob)
        count_vec = np.vectorize(count, signature='(n)->(k)')
        counter = count_vec(prediction)

        if return_prob:
            return counter.astype("float")/float(times)

        return np.argmax(counter, axis=1)


    def predict_by_groups(self, datapoint, times):
        prob = self.get_group_prob(datapoint, self.group_members_sorted)
        return [ self.sample_group_predictions(group_prob, times) for group_prob in prob]

    def predict_counterfactuals_by_groups(self, datapoint, ind, label, times):
        prob = self.get_group_prob(datapoint, self.group_members_sorted)
        predictions = [ self.sample_group_predictions(group_prob, times) for group_prob in prob]
        group, i = self.index_dict[ind]
        predictions[group] = self.sample_counterfactual_predictions(prob[group], i, label, times)
        return predictions

    def predict(self,data, times =...):
        if times is ...: times = self.n_samples
        labels = []
        for datapoint in data:
            list_group_predictions = self.predict_by_groups(datapoint, times)
            predictions = []
            for k, (g,i) in sorted(self.index_dict.items()):
                predictions.append(list_group_predictions[g][i])
            labels.append(np.asarray(predictions))
        return np.asarray(labels)
    
    def predict_counterfactuals(self, data, obs_inds, obs_labels, times = ...):
        if times is ...: times = self.n_samples

        labels = np.empty((data.shape[0], self.n_experts))
        for x in range(data.shape[0]):
            list_group_predictions = self.predict_counterfactuals_by_groups(data[x,:], obs_inds[x], obs_labels[x],times)
            predictions = np.empty((self.n_experts))
            for k, (g,i) in sorted(self.index_dict.items()):
                predictions[k]=list_group_predictions[g][i]
            labels[x] = predictions
        return labels


    def predict_cfc_proba(self, data, obs_inds, obs_labels, times = ...):
        if times is ...: times = self.n_samples
        proba = np.empty((data.shape[0], self.n_experts, self.n_classes), dtype=float)
        for x in range(data.shape[0]):
            prob = self.get_group_prob(data[x,:], self.group_members_sorted)
            list_group_predictions = [ self.sample_group_predictions(group_prob, times, return_prob=True) for group_prob in prob]
            group, i = self.index_dict[obs_inds[x]]
            list_group_predictions[group] = self.sample_counterfactual_predictions(prob[group], i, obs_labels[x], times, return_prob=True)
            predictions = np.empty((self.n_experts, self.n_classes))
            for k, (g,i) in sorted(self.index_dict.items()):
                predictions[k]=list_group_predictions[g][i]
            proba[x] = predictions
        return proba
    
    #samples naive predictions and counterfactual predictions for the data, returns avg. diff in error for each observed expert
    def get_error_diff_list(self, expert_list, data, labels):
        #sample naive(i.e., marginal) error
        predictions = np.empty_like(labels)
        print('estimating naive error')
        for x in range(data.shape[0]):
            start = perf_counter()
        
            all_prob = np.vstack(self.get_prob(data[x]))
            sample_vec = np.vectorize(self.sample_group_predictions, signature='(m,n)->(m)')
            d_predictions = sample_vec(all_prob)
            predictions[x] = np.transpose(d_predictions)

            duration =  (perf_counter() - start)
            if x==0: print("time to estimate naive error for datapoint ", x, " : ", duration, "s")
        
        loss = np.not_equal(predictions, labels)
        #set error to nan if no data
        error_naive = np.mean(loss, 0, where=labels != -999)
        print('done estimating naive error')

        #sample counterfactual error
        error_cf_list = np.empty((self.n_experts, self.n_experts))
        print('estimating gumbel cscm error')
        for obs_ind in range(self.n_experts):
          start = perf_counter()
          if obs_ind in expert_list:
            #sample counterfactual labels given observed expert obs_ind
            #note: we pretend that all experts are one PCS group for the sampling
            cf_sample_given_obs = np.vectorize(lambda d, obs_label: self.sample_counterfactual_predictions(np.vstack(self.get_prob(d)), obs_ind, obs_label), signature= '(m),()->(n)')
            obs_labels = labels[:,obs_ind]
            obs_label_index = obs_labels != -999
            obs_labels = obs_labels[obs_label_index]
            if np.size(obs_labels) != 0:
                cf_predictions = cf_sample_given_obs(data[obs_label_index], obs_labels)
                #compute error of the model given this observed expert
                true_labels = labels[obs_label_index]
                loss = np.not_equal(cf_predictions, true_labels)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    error_cf_list[obs_ind]=np.mean(loss, 0, where= true_labels != -999)
            else:
                #set error to nan if no data
                error_cf_list[obs_ind] = np.full((self.n_experts), np.nan)
          else:
                #set error to nan if no data
                error_cf_list[obs_ind] = np.full((self.n_experts), np.nan)
          duration =  (perf_counter() - start)
          if obs_ind % 100 ==0 : print("time to estimate cf error for obs. expert ", obs_ind, " : ", duration, "s")

        print('done estimating gumbel cscm error')


        # return difference in error from sampling naively to counterfactually
        # error_diff_list[obs_ind][pred_ind] = error diff. (0/1-loss) of the counterfactual prediction of pred_ind's label given obs_ind's label to the naive prediction
        return error_cf_list - error_naive

    def fit(self, data, labels, val_ratio=0.0, max_rounds=5):
        # partition into training and validation set

        # Uncomment for validation
        val_size = math.ceil(val_ratio * data.shape[0])
        #print("Validation data size for greedy alg.: ", val_size)
        val_data, data = np.vsplit(data, [val_size])
        val_labels, labels = np.vsplit(labels, [val_size])
        #print(val_labels)
        # run algorithm to find best grouping (clique partition)
        #construct Graph from data and prob. functions
        start = perf_counter()
        self.graph = PCS_graph(self.list_prob_functions)
        self.graph.resolve_edges(data, labels)
        duration =  (perf_counter() - start)
        print("time to resolve edges: ", duration, "s")

        expert_list = self.graph.get_nodes_with_edges()
        best_partition = []
        if len(expert_list)>0:
            start = perf_counter()
            train_error_diff_list = self.get_error_diff_list(expert_list,data,labels)
            if val_ratio== 0.0:
                val_error_diff_list = train_error_diff_list
            else:
                val_error_diff_list = self.get_error_diff_list(expert_list,val_data,val_labels)
            #set graph weights
            self.graph.set_training_weights(train_error_diff_list)
            self.graph.set_validation_weights(val_error_diff_list)
            duration =  (perf_counter() - start)
            print("time to estimate errors : ", duration, "s")

            start = perf_counter()
            #run fit to find best partition

            # Uncomment for validation
            best_partition = self.graph.fit(max_rounds)
            duration =  (perf_counter() - start)
            print("greedy algorithm runtime : ", duration, "s")

        else:
            #if no edges set naive as best_partition
            best_partition = [ set([ind]) for ind in range(self.n_experts)] 
        #set group list and dictionary
        self.set_group_membership(best_partition)

    def analyze_PCS_graph(self, data, labels):
        #construct Graph from data and prob. functions
        self.graph = PCS_graph(self.list_prob_functions)
        self.graph.resolve_edges(data, labels)
        return self.graph.analyze(self.group_members_sorted)


    def score_counterfactuals_rand(self, data, labels, test_inds, test_labels):
        test_predictions = np.zeros((data.shape[0]))
        for x in range(data.shape[0]):
            log_prob = self.get_group_prob(data[x,:], self.group_members_sorted)
            group, i = self.index_dict[test_inds[x]]
            obs_group = [j for j in self.group_members_sorted[group] if j!=test_inds[x] and labels[x,j]!=-999]
            group_size = len(obs_group)
            if group_size ==0:
                test_predictions[x] = self.sample_group_predictions(log_prob[group])[i]
            else:
                obs_ind = self.rng.choice(a=obs_group)
                group_obs, obs_ind_group = self.index_dict[obs_ind]
                if group != group_obs: "something is wrong"
                group_predictions = self.sample_counterfactual_predictions(log_prob[group], obs_ind_group, labels[x, obs_ind])
                test_predictions[x] = group_predictions[i]

        loss_matrix = np.not_equal(test_predictions, test_labels)
        return np.mean(loss_matrix)

    def save(self):
        self.graph.save(self.name, self.group_members)


