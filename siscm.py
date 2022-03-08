#!/usr/bin/env python

import numpy as np
from PCS_graph import PCS_graph
import math 
from sklearn.metrics import top_k_accuracy_score

import warnings
from time import perf_counter


class SISCM:
    list_marginal_proba_func = []

    def __init__(self, name, n_classes,  list_proba_functions, group_members = [], siscm_H = False, n_samples =1000):
        self.name = name
        self.n_classes = n_classes
        self.n_samples = n_samples
        #experts' marginal probability functions
        self.list_marginal_proba_func = list_proba_functions
        self.n_experts = len(list_proba_functions)
        #Gumbel Max parameters
        self.mu, self.beta = 0, 1
        self.rng = np.random.default_rng(42)
        # PCS graph for training
        self.graph = None

        #optional when groups are know (e.g., sampling synthetic data)
        if siscm_H:
            group_members = [ set([ind]) for ind in range(self.n_experts)] 
        self.set_group_membership(group_members)

    #helper functions
    #set group membership
    #input: list of groups (sets) of experts
    def set_group_membership(self, group_membership):
        self.group_members = group_membership #list of sets (set = group)
        self.group_members_sorted = [ sorted(group) for group in group_membership] #list of sorted lists (sorted list = group)
        print("Partition of Model ", self.name)
        print(self.group_members_sorted)
        self.n_groups= len(group_membership)

        # construct index dictionary from group members list
        index_dict = {}
        for g in range(len(self.group_members_sorted)):
            id = 0
            for index in self.group_members_sorted[g]:
                index_dict[index] = g, id #associates overall expert id (index) with group index and (sorted) position () inside the group
                id+=1
        self.group_membership_dict = index_dict

    #returns all members (set) of group this individual belongs to
    def get_group(self, ind):
        return self.group_members[self.group_membership_dict[ind][0]]

    #returns index of group this individual belong
    def get_group_index(self, ind):
        return self.group_membership_dict[ind][0]

    #return group_members list
    def get_group_members(self):
        return self.group_members

    #return experts probabilitity functions
    def get_proba_function(self):
        return self.list_marginal_proba_func

    #compute individual experts probabilities for datapoint
    def get_proba(self, datapoint):
        #return [ np.asarray([np.log(f(datapoint)) for f in self.list_proba_functions])]
        return [ np.asarray([f(datapoint) for f in self.list_marginal_proba_func])]

    #compute experts probabilities for datapoint, returns it as a list of group probability matrices
    def get_group_proba(self, datapoint, group_membership_sorted):
        group_proba = []
        for g in range(len(group_membership_sorted)):
            list_proba = [ self.list_marginal_proba_func[ind](datapoint) for ind in group_membership_sorted[g]]
            group_proba.append( np.asarray( list_proba ))
        return group_proba

    #sample group predictions from using gumbels sampled from pior
    #group_proba : matrix n_experts (in group) x n_classes yy
    def sample_predictions_by_group(self, group_proba, n_samples=..., return_proba=False):
        #set number of samples
        if n_samples is ...: n_samples = self.n_samples
        #counts occurences of each class label in predictions
        def count(pred_ind):
            val, count = np.unique(pred_ind, return_counts=True)
            full_count = np.zeros((self.n_classes))
            full_count[val] = count
            return full_count

        #sample gumbels
        u = self.rng.gumbel(self.mu, self.beta, size=(n_samples, self.n_classes))
        argmax = np.vectorize((lambda p: np.argmax( np.log(p) + u, axis=1)), signature='(n)->(t)')
        pred_per_sample = argmax( group_proba)  #likest prediction per expert per sample (n_experts x n_samples)
        count_vec = np.vectorize(count, signature='(n)->(k)') 
        counter = count_vec(pred_per_sample) #count of occurence of each class label for each expert (n_experts x n_classes)
        
        if return_proba:
        #return probabilities for each expert (n_expert x n_classes)
            return counter.astype("float")/float(n_samples) #Monte carlo estimator of the probability

        #returns most likely prediction for each expert
        return np.argmax(counter, axis=1)

    #sample gumbels from posterior given realization s_p_real and marginal probabilities trans_probabilities
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
    #group_proba : matrix n_experts (in group) x n_classes yy
    #note g_ind is the observed individuals index in the group not overall
    def sample_cf_predictions_by_group(self, group_proba, g_ind, label, n_samples=..., return_proba=False):
        #set number of samples
        if n_samples is ...: n_samples = self.n_samples
        
        #counts occurences of each class label in predictions
        def count(pred_ind):
            val, count = np.unique(pred_ind, return_counts=True)
            full_count = np.zeros((self.n_classes))
            full_count[val] = count
            return full_count

        #sample gumbels from posterior
        u = self.sample_gumbels(group_proba[g_ind,:], label, n_samples)
        argmax = np.vectorize((lambda p: np.argmax( np.log(p) + u, axis=1)), signature='(n)->(t)')
        prediction = argmax( group_proba) #likest prediction per expert per sample (n_experts x n_samples)
        count_vec = np.vectorize(count, signature='(n)->(k)')
        counter = count_vec(prediction)#count of occurence of each class label for each expert (n_experts x n_classes)

        #return counterfactual probabilities for each expert (n_expert x n_classes)
        if return_proba:
            return counter.astype("float")/float(n_samples) #monte carlo estimator of the counterfactual probabilities

        #returns most likely counterfactual prediction for each expert
        return np.argmax(counter, axis=1)

    #sample predictions by groups
    def predict_by_groups(self, datapoint, n_samples):
        #get marginal probabilities of each expert
        proba = self.get_group_proba(datapoint, self.group_members_sorted)
        #sample predictions by group
        return [ self.sample_predictions_by_group(group_proba, n_samples) for group_proba in proba]

    #sample counterfactual predictions by groups
    def predict_cf_by_groups(self, datapoint, ind, label, n_samples):
        #get marginal probabilities of each expert
        proba = self.get_group_proba(datapoint, self.group_members_sorted)
        #sample predictions by group
        predictions = [ self.sample_predictions_by_group(group_proba, n_samples) for group_proba in proba]
        #get observed experts group index and position in group
        group, id_in_group = self.group_membership_dict[ind]
        #sample counterfactual predictions by this group
        predictions[group] = self.sample_cf_predictions_by_group(proba[group], id_in_group, label, n_samples)
        return predictions

    #sample predictions
    def predict(self,data, n_samples =...):
        if n_samples is ...: n_samples = self.n_samples
        labels = []
        #for each data point sample predictions
        for datapoint in data:
            list_group_predictions = self.predict_by_groups(datapoint, n_samples)
            #reorder predictions according to experts global id
            predictions = []
            for k, (g,i) in sorted(self.group_membership_dict.items()):
                predictions.append(list_group_predictions[g][i])
            labels.append(np.asarray(predictions))
        return np.asarray(labels)
    
    #sample counterfactual predictions
    def predict_cf(self, data, obs_inds, obs_labels, n_samples = ...):
        if n_samples is ...: n_samples = self.n_samples

        labels = np.empty((data.shape[0], self.n_experts))
        #for each data point sample predictions
        for x in range(data.shape[0]):
            list_group_predictions = self.predict_cf_by_groups(data[x,:], obs_inds[x], obs_labels[x],n_samples)
            #reorder predictions according to experts global id
            predictions = np.empty((self.n_experts))
            for k, (g,i) in sorted(self.group_membership_dict.items()):
                predictions[k]=list_group_predictions[g][i]
            labels[x] = predictions
        return labels


    #estimate counterfactual probabilities
    def predict_cfc_proba(self, data, obs_inds, obs_labels, n_samples = ...):
        if n_samples is ...: n_samples = self.n_samples
        cf_proba = np.empty((data.shape[0], self.n_experts, self.n_classes), dtype=float)
        #for each data point estimate cf probabilities
        for x in range(data.shape[0]):
            #get marginal probabilities by groups
            marginal_proba = self.get_group_proba(data[x,:], self.group_members_sorted)
            #estimate counterfactual probabilities (for experts not in observed group ~ marginal probabilities)
            list_group_proba = [ self.sample_predictions_by_group(group_proba, n_samples, return_proba=True) for group_proba in marginal_proba]

            #get observed experts group index and position in group
            group, i = self.group_membership_dict[obs_inds[x]]
            #estimate counterfactual probabilities of this group
            list_group_proba[group] = self.sample_cf_predictions_by_group(marginal_proba[group], i, obs_labels[x], n_samples, return_proba=True)
            predictions = np.empty((self.n_experts, self.n_classes))

            #reorder predictions according to experts global id
            for k, (g,i) in sorted(self.group_membership_dict.items()):
                predictions[k]=list_group_proba[g][i]
            cf_proba[x] = predictions

        return cf_proba
    
    #samples counterfactual predictions and predictions from marginal distribution model for each pair of experts on the data, returns avg. diff in error for pair
    #error_diff[obs_ind][pred_ind] = error diff. (0/1-loss) of the counterfactual prediction of pred_ind's label given obs_ind's label to the prediction from marginal distribution
    def get_error_difference(self, expert_list, data, labels):
        #sample marginal error
        predictions = np.empty_like(labels)
        print('Estimating naive error...')
        #get likeliest predictions from marginal distribution
        for x in range(data.shape[0]):
            start = perf_counter()
        
            all_proba = np.vstack(self.get_proba(data[x]))
            sample_vec = np.vectorize(self.sample_predictions_by_group, signature='(m,n)->(m)')
            d_predictions = sample_vec(all_proba)
            predictions[x] = np.transpose(d_predictions)

            duration =  (perf_counter() - start)
            if x==0: print("time to estimate naive error for datapoint ", x, " : ", duration, "s")
        #compute error
        loss = np.not_equal(predictions, labels)
        #set error to nan if no data
        marginal_error = np.mean(loss, 0, where=labels != -999) #marginal error per expert (vector of size n_experts)
        print('Done estimating naive error')

        #sample counterfactual error
        cf_error = np.empty((self.n_experts, self.n_experts)) #cf error per expert given anobserved expert
        print('Estimating gumbel cscm error')
        #compute error for each expert given an observed expert
        for obs_ind in range(self.n_experts):
          start = perf_counter()
          if obs_ind in expert_list:
            #sample counterfactual labels given observed expert obs_ind
            #note: we reuse code by pretending that all experts are one PCS group for the sampling
            #defining vectorized cf sampling function
            vec_cf_sample_given_obs = np.vectorize(lambda d, obs_label: self.sample_cf_predictions_by_group(np.vstack(self.get_proba(d)), obs_ind, obs_label), signature= '(m),()->(n)')
            #get observations
            obs_labels = labels[:,obs_ind]
            obs_label_index = obs_labels != -999
            obs_labels = obs_labels[obs_label_index]
            if np.size(obs_labels) != 0:
                #if expert has observations, sample cf predictions for other experts
                cf_predictions = vec_cf_sample_given_obs(data[obs_label_index], obs_labels)
                #compute error of the model given this observed expert
                true_labels = labels[obs_label_index]
                loss = np.not_equal(cf_predictions, true_labels)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    cf_error[obs_ind]=np.mean(loss, 0, where= true_labels != -999)
            else:
                #set error to nan if no data
                cf_error[obs_ind] = np.full((self.n_experts), np.nan)
          else:
                #set error to nan if no data
                cf_error[obs_ind] = np.full((self.n_experts), np.nan)
          duration =  (perf_counter() - start)
          if obs_ind % 100 ==0 : print("time to estimate cf error for obs. expert ", obs_ind, " : ", duration, "s")

        print('Done estimating gumbel cscm error')


        # return difference in error from sampling naively to counterfactually
        error_diff = cf_error - marginal_error
        return error_diff

    #fit model using training data:
    #Finding a partition od the experts into mutually similar expert groups
    def fit(self, data, labels, val_ratio=0.0, max_rounds=5):
        #partition into training and validation set
        #note validation set not used for training only to compare goodness of the expert partition
        #when val_ratio=0.0 we compare overall error reduction on the whole training data
        val_size = math.ceil(val_ratio * data.shape[0])
        #print("Validation data size for greedy alg.: ", val_size)
        val_data, data = np.vsplit(data, [val_size])
        val_labels, labels = np.vsplit(labels, [val_size])
        #run algorithm to find best grouping (clique partition)
        #construct Graph from data and proba. functions
        start = perf_counter()
        self.graph = PCS_graph(self.list_marginal_proba_func)
        #check conditional stability conditions
        self.graph.resolve_edges(data, labels)
        duration =  (perf_counter() - start)
        print("time to resolve edges: ", duration, "s")

        #get list of experts with edges
        expert_list = self.graph.get_nodes_with_edges()
        best_partition = []
        if len(expert_list)>0:
            start = perf_counter()
            #get error difference matrix
            train_error_diff = self.get_error_difference(expert_list,data,labels)
            if val_ratio== 0.0:
                val_error_diff = train_error_diff
            else:
                val_error_diff = self.get_error_difference(expert_list,val_data,val_labels)
            #set graph weights
            self.graph.set_training_weights(train_error_diff)
            self.graph.set_validation_weights(val_error_diff)
            duration =  (perf_counter() - start)
            print("time to estimate errors : ", duration, "s")

            start = perf_counter()
            #run clique partition algorithm to find best partition
            best_partition = self.graph.fit(max_rounds)
            duration =  (perf_counter() - start)
            print("greedy algorithm runtime : ", duration, "s")

        else:
            #if no edges set naive as best_partition
            best_partition = [ set([ind]) for ind in range(self.n_experts)] 
        #set new group membership
        self.set_group_membership(best_partition)

    #analyse edge ratio of the conditional stability graph
    def analyze_PCS_graph(self, data, labels):
        #construct Graph from data and proba. functions
        self.graph = PCS_graph(self.list_marginal_proba_func)
        self.graph.resolve_edges(data, labels)
        return self.graph.analyze_edge_ratio(self.group_members_sorted)

    #scoring function for synthetic experiments
    #samples cf predictions for test experts returns error incurred by the model
    def score_counterfactuals_rand(self, data, labels, test_inds, test_labels):
        test_predictions = np.zeros((data.shape[0]))
        #sample cf predictions for each test pair (datapoint, expert h) by sampling an expert to observe from the group of expert h
        for x in range(data.shape[0]):
            log_proba = self.get_group_proba(data[x,:], self.group_members_sorted)
            group, i = self.group_membership_dict[test_inds[x]]
            obs_group = [j for j in self.group_members_sorted[group] if j!=test_inds[x] and labels[x,j]!=-999]
            group_size = len(obs_group)
            if group_size ==0:
                test_predictions[x] = self.sample_predictions_by_group(log_proba[group])[i]
            else:
                obs_ind = self.rng.choice(a=obs_group)
                group_obs, obs_ind_group = self.group_membership_dict[obs_ind]
                if group != group_obs: "something is wrong"
                group_predictions = self.sample_cf_predictions_by_group(log_proba[group], obs_ind_group, labels[x, obs_ind])
                test_predictions[x] = group_predictions[i]

        loss_matrix = np.not_equal(test_predictions, test_labels)
        return np.mean(loss_matrix)

    #save graph
    def save(self):
        self.graph.save(self.name, self.group_members)


