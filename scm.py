#!/usr/bin/env python

import numpy as np
import random
from PCS_graph import PCS_graph
import math 
from sklearn.metrics import top_k_accuracy_score

import warnings
from numba import njit, prange
import multiprocessing as mup
import psutil
from functools import partial
from time import perf_counter


class SCM:
    list_prob_functions = []
    n_samples = 100 #50
    n_failed_r_sampling = 0

    def __init__(self, name, n_classes, list_prob_functions, group_members = [], naive = False):
        self.name = name
        self.n_classes = n_classes
        #experts' marginal probability functions
        self.list_prob_functions = list_prob_functions
        self.n_experts = len(list_prob_functions)
        #Gumbel Max parameters
        self.mu, self.beta = 0, 1
        self.rng = np.random.default_rng(42)
        #self.threads_used = psutil.cpu_count(logical=True)-1
        #print(self.threads_used, 'threads used')
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
    #TO DO: have group_members as argument
    def get_group_prob(self, datapoint, group_membership):
        group_prob = []
        for g in range(len(group_membership)):
            #list_log_prob = [ np.log(self.list_prob_functions[ind](datapoint)) for ind in group_membership[g]]

            list_log_prob = [ self.list_prob_functions[ind](datapoint) for ind in group_membership[g]]
            group_prob.append( np.asarray( list_log_prob ))
        return group_prob

    #sample group prediction from prior
    """
    def sample_group_predictions(self, group_prob, times=n_samples):
        u = self.rng.gumbel(self.mu, self.beta, self.n_classes)
        argmax = np.vectorize((lambda log_p: np.argmax( log_p + u)), signature='(n)->()')
        return argmax( group_prob)
    """
    def sample_group_predictions(self, group_prob, times=n_samples, return_prob=False):
        #counter = np.zeros((self.n_classes, group_prob.shape[0]))
        #print("sample")
        #print(group_prob.shape)
        #rng = np.random.default_rng(seed+1)
        """
            for t in range(times):
            #u = self.rng.gumbel(self.mu, self.beta, self.n_classes)
            #argmax = np.vectorize((lambda log_p: np.argmax( log_p + u)), signature='(n)->()')
            prediction_matrix = np.zeros((self.n_classes, group_prob.shape[0]))
            prediction_matrix[prediction,range(group_prob.shape[0])] = 1
            #print(prediction)
            #print(prediction_matrix)
            counter = counter + prediction_matrix 
        """
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
            return counter.astype("float")/times
        
        return np.argmax(counter, axis=1)

    #sample group prediction from posterior
    #note g_ind is the observed individuals index in the group not overall
    """
    def sample_counterfactual_predictions(self, group_prob, g_ind, label, times=n_samples):
        posterior_noise = False
        max_iter = 0
        #print(g_ind)
        #print(label)
        #print(group_prob[g_ind, :])
        while(not posterior_noise):
            max_iter +=1
            u = self.rng.gumbel(self.mu, self.beta, self.n_classes)
            posterior_noise = (label == np.argmax( group_prob[g_ind,:] + u))
            if max_iter > 10000: 
                self.n_failed_r_sampling +=1
                break

        argmax = np.vectorize((lambda log_p: np.argmax( log_p + u)), signature='(n)->()')
        return argmax( group_prob)
    """
    def sample_gumbels(trans_probabilities, s_p_real, num_of_samples):
        
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
        gumbels = [topdown(trans_probabilities, s_p_real, np.random.default_rng(seed+1)) for seed in range(num_of_samples)]

        # Sanity check
        for gum in gumbels:
            temp = gum + np.log(trans_probabilities)
            assert np.argmax(temp)==s_p_real, "Sampled gumbels don't match with realized argmax"
        
        return gumbels
    """
    #function for multithreaded call
    def sample_gumbels(trans_probabilities, s_p_real, seed):

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
        gumbels = topdown(trans_probabilities, s_p_real, np.random.default_rng(seed+1))

        # Sanity check
        temp = gumbels + np.log(trans_probabilities)
        assert np.argmax(temp)==s_p_real, "Sampled gumbels don't match with realized argmax"
            
        return gumbels
    """
    def sample_counterfactual_predictions(self, group_prob, g_ind, label, times=n_samples, return_prob=False):
        """
        counter = np.zeros((self.n_classes, group_prob.shape[0]))
        p = mup.Pool(processes=self.threads_used)
        gumbels = p.map(partial(SCM.sample_gumbels, group_prob[g_ind,:], label), range(g_ind, g_ind+times))
        p.close()
        for gumbel_noise in gumbels:
            #argmax = np.vectorize((lambda log_p: np.argmax( log_p + gumbel_noise)), signature='(n)->()')
            argmax = np.vectorize((lambda p: np.argmax( np.log(p) + gumbel_noise)), signature='(n)->()')
            prediction = argmax( group_prob)
            prediction_matrix = np.zeros((self.n_classes, group_prob.shape[0]))
            prediction_matrix[prediction,range(group_prob.shape[0])] = 1
            #print(prediction)
            #print(prediction_matrix)
            counter = counter + prediction_matrix 
        return np.argmax(counter, axis=0)
        """
        def count(pred_ind):
            val, count = np.unique(pred_ind, return_counts=True)
            full_count = np.zeros((self.n_classes))
            full_count[val] = count
            return full_count

        u = SCM.sample_gumbels(group_prob[g_ind,:], label, times)
        argmax = np.vectorize((lambda p: np.argmax( np.log(p) + u, axis=1)), signature='(n)->(t)')
        prediction = argmax( group_prob)
        count_vec = np.vectorize(count, signature='(n)->(k)')
        counter = count_vec(prediction)

        if return_prob:
            return counter.astype("float")/times

        return np.argmax(counter, axis=1)


    """
    def sample_counterfactual_predictions(self, group_prob, g_ind, label, times=n_samples):
        counter = np.zeros((self.n_classes, group_prob.shape[0]))
        #print("sample")
        #print(group_prob.shape)
        for t in range(times):
            posterior_noise = False
            max_iter = 0
            while(not posterior_noise):
                max_iter +=1
                u = self.rng.gumbel(self.mu, self.beta, self.n_classes)
                posterior_noise = (label == np.argmax( group_prob[g_ind,:] + u))
                if max_iter > 1000: 
                    self.n_failed_r_sampling +=1
                    break
            argmax = np.vectorize((lambda log_p: np.argmax( log_p + u)), signature='(n)->()')
            prediction = argmax( group_prob)
            prediction_matrix = np.zeros((self.n_classes, group_prob.shape[0]))
            prediction_matrix[prediction,range(group_prob.shape[0])] = 1
            #print(prediction)
            #print(prediction_matrix)
            counter = counter + prediction_matrix 
        return np.argmax(counter, axis=0)
    """

    def predict_by_groups(self, datapoint, times):
        prob = self.get_group_prob(datapoint, self.group_members_sorted)
        return [ self.sample_group_predictions(group_prob, times) for group_prob in prob]

    def predict_counterfactuals_by_groups(self, datapoint, ind, label, times):
        prob = self.get_group_prob(datapoint, self.group_members_sorted)
        predictions = [ self.sample_group_predictions(group_prob, times) for group_prob in prob]
        group, i = self.index_dict[ind]
        predictions[group] = self.sample_counterfactual_predictions(prob[group], i, label, times)
        return predictions

    def predict(self,data, times =n_samples):
        labels = []
        for datapoint in data:
            list_group_predictions = self.predict_by_groups(datapoint, times)
            predictions = []
            for k, (g,i) in sorted(self.index_dict.items()):
                predictions.append(list_group_predictions[g][i])
            labels.append(np.asarray(predictions))
        return np.asarray(labels)
    '''
    def predict_counterfactuals(self, data, obs_inds, obs_labels, times = n_samples):
        labels = []
        for x in range(data.shape[0]):
            list_group_predictions = self.predict_counterfactuals_by_groups(data[x,:], obs_inds[x], obs_labels[x],times)
            predictions = []
            for k, (g,i) in sorted(self.index_dict.items()):
                predictions.append(list_group_predictions[g][i])
            labels.append(np.asarray(predictions))
        return np.asarray(labels)
    '''
    def predict_counterfactuals(self, data, obs_inds, obs_labels, times = n_samples):
        labels = np.empty((data.shape[0], self.n_experts))
        for x in range(data.shape[0]):
            list_group_predictions = self.predict_counterfactuals_by_groups(data[x,:], obs_inds[x], obs_labels[x],times)
            predictions = np.empty((self.n_experts))
            for k, (g,i) in sorted(self.index_dict.items()):
                predictions[k]=list_group_predictions[g][i]
            labels[x] = predictions
        return labels


    def predict_cfc_proba(self, data, obs_inds, obs_labels, times = n_samples):
        proba = np.empty((data.shape[0], self.n_experts, self.n_classes))
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
            """
            inputs = zip(self.get_prob(d),range(self.n_experts))
            p = mup.Pool(processes=self.threads_used)
            x_prediction = p.starmap(self.sample_group_predictions, inputs)
            predictions.append(np.vstack(x_prediction.get()))
            p.close()
            """
            start = perf_counter()
        
            all_prob = np.vstack(self.get_prob(data[x]))
            sample_vec = np.vectorize(self.sample_group_predictions, signature='(m,n)->(m)')
            d_predictions = sample_vec(all_prob)
            predictions[x] = np.transpose(d_predictions)

            duration =  (perf_counter() - start)
            if x==1: print("time to estimate naive error for datapoint ", x, " : ", duration, "s")
            """
            all_prob = self.get_prob(data[x])
            d_predictions = np.vstack([self.sample_group_predictions(ind_prob) for ind_prob in all_prob])
            predictions.append(d_predictions)
            """
            """
            if x == 1:
                print("Prob:")
                print(all_prob)
                print("Pred:")
                print(d_predictions)
                print("Label:")
                print(labels[x])
            """

        # for handling missing values:
        loss = np.not_equal(predictions, labels)
        #loss[labels == -999] = False
        #set error to nan if no data
        error_naive = np.mean(loss, 0, where=labels != -999)
        print('done estimating naive error')

        #sample counterfactual error
        error_cf_list = np.empty((self.n_experts, self.n_experts))
        #no_label_vertex_set=set()
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
                #loss[true_labels == -999] = False
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
        #return (np.vstack([ error - error_naive for error in error_cf_list]), no_label_vertex_set)
        return error_cf_list - error_naive
        

    """
    def get_errDiff(self, candidates, prevDiff, newnode, error_diff_list):
        newDiff = {}
        for c in candidates:
            loss_subgroup, loss_candidate = prevDiff[c]
            newDiff[c] = (loss_subgroup + error_diff_list[c][newnode], loss_candidate +  error_diff_list[newnode][c])
        return newDiff


    def greedy_clique_cover(self, data, labels):
        cover = []
        error_diff_list, no_label_vertex_set = self.get_error_diff_list(data, labels)
        #print(error_diff_list)

        #greedy algorithm
        #cover.extend([{v} for v in no_label_vertex_set])
        #remaining_nodes = self.graph.get_nodes().difference(no_label_vertex_set)
        remaining_nodes = self.graph.get_nodes()
        while len(remaining_nodes):
            v = self.rng.choice(list(remaining_nodes))
            remaining_nodes.remove(v)
            #v = remaining_nodes.pop()
            subgroup = set([v])
            candidates = remaining_nodes.intersection(self.graph.get_neighbors(v))
            argmin = v
            errDiff = dict.fromkeys(list(candidates), (0,0))
            while len(candidates) != 0:
                errDiff = self.get_errDiff(candidates, errDiff, argmin, error_diff_list)
                #print(errDiff)
                #TO DO: should we stop if the error increases compared to naive ???
                if all(sum(val)>0 for val in errDiff.values()): break
                # add to argmin to subgroup
                argmin = min(errDiff, key = lambda k: sum(errDiff[k]))#/(len(subgroup)+1))
                subgroup.add(argmin)
                # update candidates and errDiff by removing non neighbors of argmin
                neighbors = self.graph.get_neighbors(argmin)
                diff = candidates - neighbors 
                for non_candidates in diff:
                    errDiff.pop(non_candidates, None)
                candidates.intersection_update(neighbors)

            #add new subgroup, remove nodes from the graph
            cover.append(subgroup)
            remaining_nodes.difference_update(subgroup)
        #print(cover)
        return cover
    """
    """
    def fit(self, data, labels, val_ratio=0.2, max_rounds=5):
        #construct Graph from data and prob. functions
        self.graph = PCS_graph(self.list_prob_functions)
        self.graph.fit(data, labels)
        val_size = math.ceil(val_ratio * data.shape[0])
        print(val_size)
        val_data, data = np.vsplit(data, [val_size])
        val_labels, labels = np.vsplit(labels, [val_size])
        print(val_labels)
        # run algorithm to find best grouping (clique cover)
        test_inds = np.array([self.rng.choice(a=[ j for j in range(self.n_experts) if val_labels[i,j]!=-999]) for i in range(val_size)])
        print(test_inds)
        test_labels = val_labels[range(val_size),test_inds]
        print(test_labels)
        best_group = None
        best_score = 1
        #find best cover through validation set
        for rounds in range(max_rounds):
            group_members = self.greedy_clique_cover(data, labels)
            self.set_group_membership(group_members)
            score = self.score_counterfactuals_rand(val_data, val_labels, test_inds, test_labels)
            print(score)
            if score <= best_score:
                best_group = group_members
                best_score = score

        #set group list and dictionary
        self.set_group_membership(best_group)
    """
    """
    def fit(self, data, labels, val_ratio=0.2, max_rounds=5):
        #construct Graph from data and prob. functions
        self.graph = PCS_graph(self.list_prob_functions)
        self.graph.check_PCS_condition(data, labels)
        # partition into training and validation set
        val_size = math.ceil(val_ratio * data.shape[0])
        print(val_size)
        val_data, data = np.vsplit(data, [val_size])
        val_labels, labels = np.vsplit(labels, [val_size])
        print(val_labels)
        # run algorithm to find best grouping (clique cover)
        best_group = None
        best_score = math.inf
        error_diff_list, no_label_vertex_set = self.get_error_diff_list(val_data,val_labels)
        #find best cover through validation set
        for rounds in range(max_rounds):
            group_members = self.greedy_clique_cover(data, labels)
            score = 0
            for g in group_members:
                expert_set_with_data = np.asarray(list(g.difference(no_label_vertex_set)))
                print(expert_set_with_data)
                if expert_set_with_data.size > 0:
                    score = score + np.nansum(np.vstack(error_diff_list)[expert_set_with_data[:,np.newaxis], expert_set_with_data])

            print(score)
            if score <= best_score:
                best_group = group_members
                best_score = score

        #set group list and dictionary
        self.set_group_membership(best_group)
    """
    def fit(self, data, labels, val_ratio=0.2, max_rounds=5):
        # partition into training and validation set

        # Uncomment for validation
        val_size = 0 #math.ceil(val_ratio * data.shape[0])
        print("Validation data size for greedy alg.: ", val_size)
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

        #train_error_diff_list, train_no_label_vertex_set = self.get_error_diff_list(data,labels)
        #val_error_diff_list, val_no_label_vertex_set = self.get_error_diff_list(val_data,val_labels)
        expert_list = self.graph.get_nodes_with_edges()
        best_partition = []
        if len(expert_list)>0:
            start = perf_counter()
            train_error_diff_list = self.get_error_diff_list(expert_list,data,labels)
            val_error_diff_list = self.get_error_diff_list(expert_list,val_data,val_labels)
            #set graph weights
            self.graph.set_training_weights(train_error_diff_list)
            self.graph.set_validation_weights(val_error_diff_list)
            duration =  (perf_counter() - start)
            print("time to estimate errors : ", duration, "s")

            start = perf_counter()
            #run fit to find best partition

            # Uncomment for validation
            best_partition = self.graph.fit(1)#max_rounds)
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

    def score(self, data, labels):
        predictions = self.predict(data)
        return np.mean(np.not_equal(predictions, labels))

    def score_counterfactuals(self, data, labels, obs_inds, obs_labels):
        predictions = self.predict_counterfactuals(data, obs_inds, obs_labels)
        out = np.full_like(labels, -999)
        same_pred = np.repeat(np.expand_dims(obs_labels, axis=1),self.n_experts,axis=1)
        loss_same_pred = np.not_equal(same_pred, labels, out=out, where= labels!=-999)
        print("Mean loss for same pred estimation: ", np.mean(loss_same_pred, where= labels!=-999))
        loss = np.not_equal(predictions, labels, out=out, where= labels!=-999)
        cf_non_same_right = np.not_equal(predictions, same_pred) * np.equal(predictions, labels)
        cf_non_same_wrong = np.logical_and(np.not_equal(predictions, same_pred), loss==1)
        print(self.name, " evaluation:")
        print("Number different and correct predictions", np.sum(cf_non_same_right))
        print("Number different and wrong predictions", np.sum(cf_non_same_wrong))
        cf_same_wrong = np.logical_and(np.equal(predictions, same_pred), loss==1) 
        print("Number same and wrong predictions", np.sum(cf_same_wrong))
        return np.mean(loss, where= labels !=-999)

    def score_counterfactuals_top_k(self, k, data, labels, obs_inds, obs_labels):
        predictions = self.predict_cfc_proba(data, obs_inds, obs_labels)
        #for any expert ind with label!=-999
        #check if labels in top k
        scores = np.empty((self.n_experts))
        for ind in range(self.n_experts):
            pred_proba = predictions[:, ind,:]
            seen_idx = labels[:, ind]!=-999
            if np.any(seen_idx):
                scores[ind]= top_k_accuracy_score(y_true=labels[seen_idx,ind], y_score=pred_proba[seen_idx], k=k, labels=np.arange(self.n_classes))
            else:
                scores[ind] = np.nan
        return np.nanmean(scores)

    def score_counterfactuals_rand(self, data, labels, test_inds, test_labels):
        naive_counter = 0
        cf_non_same_right = 0
        cf_non_same_wrong = 0
        cf_same_wrong = 0
        loss_non_naive = np.zeros((data.shape[0]))
        loss_same_pred = np.zeros((data.shape[0]))
        all_loss_same_pred = np.zeros((data.shape[0]))
        test_predictions = np.zeros((data.shape[0]))
        for x in range(data.shape[0]):
            log_prob = self.get_group_prob(data[x,:], self.group_members_sorted)
            group, i = self.index_dict[test_inds[x]]
            obs_group = [j for j in self.group_members_sorted[group] if j!=test_inds[x] and labels[x,j]!=-999]
            group_size = len(obs_group)
            if group_size ==0:
                test_predictions[x] = self.sample_group_predictions(log_prob[group])[i]
                naive_counter +=1
                obs_group_all = [j for j in range(self.n_experts) if j!=test_inds[x] and labels[x,j]!=-999]
                if len(obs_group_all)>0:
                    obs_ind_rand = self.rng.choice(a=obs_group_all)
                    all_loss_same_pred[x]=labels[x,obs_ind_rand]!=test_labels[x]
            else:
                obs_ind = self.rng.choice(a=obs_group)
                group_obs, obs_ind_group = self.index_dict[obs_ind]
                if group != group_obs: "something is wrong"
                group_predictions = self.sample_counterfactual_predictions(log_prob[group], obs_ind_group, labels[x, obs_ind])
                test_predictions[x] = group_predictions[i]
                loss_non_naive[x] = test_predictions[x]!=test_labels[x]
                loss_same_pred[x] = labels[x,obs_ind]!=test_labels[x]
                if test_predictions[x]==test_labels[x] and test_predictions[x]!=labels[x,obs_ind]:
                    cf_non_same_right +=1
                elif test_predictions[x]!=test_labels[x] and test_predictions[x]!=labels[x,obs_ind]:
                    cf_non_same_wrong +=1
                elif test_predictions[x]!=test_labels[x] and test_predictions[x]==labels[x,obs_ind]:
                    cf_same_wrong +=1

        loss_matrix = np.not_equal(test_predictions, test_labels)
        with warnings.catch_warnings():
            #ignore empty mean warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            print("Number of naive inferences: ", naive_counter)
            print("Number of non same predictions correct: ", cf_non_same_right)
            print("Number of non same predictions wrong: ", cf_non_same_wrong)
            print("Number of same predictions wrong: ", cf_same_wrong)
            print("Mean loss for non naive estimation: ", np.mean(loss_non_naive))
            print("Mean loss for same pred estimation: ", np.mean(loss_same_pred))
            print("Mean loss for same pred estimation on all test data: ", np.mean(loss_same_pred +all_loss_same_pred))
        return np.mean(loss_matrix)

    def save(self):
        self.graph.save(self.name, self.group_members)

    #def print_number_failed_attemps(self):
    #    print("Failed rejection sampling attemps: ", self.n_failed_r_sampling)

