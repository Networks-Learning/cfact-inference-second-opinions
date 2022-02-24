#!/usr/bin/env python

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#from pyvis.network import Network
import math
import warnings
#import numba
#from numba import njit, prange
from time import perf_counter

class PCS_graph:
    def __init__(self, prob_functions):
        self.rng = np.random.default_rng(42)
        self.n_nodes = len(prob_functions)
        self.list_prob_functions = prob_functions
        self.gdict = {}
        #for v in range(self.n_nodes):
            #self.gdict[v] = set(list(range(self.n_nodes)))
            #self.gdict[v].discard(v)
        self.edges = None
        self.training_weights = None
        self.val_weights = None

    def get_nodes(self):
        return set(range(self.n_nodes))

    def get_neighbors(self, v):
        return self.gdict[v]

    def get_nodes_with_edges(self):
        node_list = []
        for node, edges in self.gdict.items():
            if len(edges)>0: node_list.append(node)
        
        return node_list

    def set_training_weights(self, directed_weights):
        self.training_weights = directed_weights + directed_weights.T

        self.training_weights[self.edges==0] = np.nan
        #print(self.edges)
        #print(self.training_weights)

        #check no Nan weights for existing edges
        assert np.isfinite(self.training_weights[self.edges==1]).all(), "existing edge with Nan training weight"
        #check Nan weights for non-existing edges
        assert np.isnan(self.training_weights[self.edges==0]).all(), "non edge with finite training weight"

    def set_validation_weights(self, directed_weights):
        self.val_weights = directed_weights + directed_weights.T
        self.val_weights[self.edges==0] = np.nan
        #print("Validation weights")
        #print(self.val_weights)

    def check_PCS_condition(n_nodes, prob_matrix, labels):
        edges = np.full((n_nodes, n_nodes), 1)
        for i in range(n_nodes):
                #check that PCS condition is satisfied
                ci= labels[i]
                if ci != -999:
                    predictions = labels
                    is_observed = (predictions!=-999)
                    predictions = predictions * (1*is_observed)
                    #to avoid division by 0, rearranged terms to use multiplication
                    mult_j = prob_matrix[np.arange(n_nodes), predictions] * prob_matrix[i,ci]
                    mult_i =  prob_matrix[:, ci] * (prob_matrix[i])[predictions]
                    is_larger = (mult_i >= mult_j) * (predictions != ci) * is_observed
                    edges[i] *= ~is_larger

        return edges


    def resolve_edges(self, data, labels):
        #function for checking PCS condition given the data and marginal prob. functions

        edges = np.ones((self.n_nodes,self.n_nodes))
        for i in range(self.n_nodes):
            has_data = labels[labels[:,i] != -999] != -999
            edges[i,:] = np.sum(has_data, axis=0) !=0
        
        np.fill_diagonal(edges, 0)

        print("number of edges before PCS: ")
        print(np.sum(edges))
        
        for x in range(data.shape[0]):
            start = perf_counter()
            #print(self.list_prob_functions[175](data[x]))
            prob = [f(data[x]) for f in self.list_prob_functions]
            #print(prob[175])
            prob_matrix = np.vstack(prob)
            #print(prob_matrix)
            #print(prob_matrix.shape)
            edges *= PCS_graph.check_PCS_condition(self.n_nodes, prob_matrix, labels[x])
            duration =  (perf_counter() - start)
            if x==1: print("time to check PCS for datapoint 1 : ", duration, "s")

        print("number of edges after PCS: ")
        print(np.sum(edges))

        self.edges = edges
        for i in range(self.n_nodes):
            self.gdict[i] = set([j for j in range(self.n_nodes) if edges[i,j]==1])


    def analyze(self, group_members_sorted):
        
        n_edges_in_groups = 0
        for g in group_members_sorted:
            n_edges_in_groups += np.sum(self.edges[np.array(g)[:, np.newaxis], g])

        print("number of edges in groups: ")
        print(n_edges_in_groups)

        return (n_edges_in_groups/np.sum(self.edges))

    def greedy_clique_partition(self):
        partition = []

        #greedy algorithm for clique partition
        remaining_nodes = self.get_nodes()
        while len(remaining_nodes):
            v = self.rng.choice(list(remaining_nodes))
            remaining_nodes.remove(v)
            #v = remaining_nodes.pop()
            subgroup = set([v])
            candidates = remaining_nodes.intersection(self.get_neighbors(v))
            argmin = v
            errDiff = dict.fromkeys(list(candidates), 0)
            while len(candidates) != 0:
                errDiff = {h: errDiff[h] + self.training_weights[h, argmin] for h in candidates}
                #print(errDiff)
                if all(val>0 for val in errDiff.values()): break
                # add to argmin to subgroup
                argmin = min(errDiff, key = lambda h: errDiff[h])
                subgroup.add(argmin)
                # update candidates and errDiff by removing non neighbors of argmin
                neighbors = self.get_neighbors(argmin)
                diff = candidates - neighbors 
                for non_candidates in diff:
                    errDiff.pop(non_candidates, None)
                candidates.intersection_update(neighbors)

            #add new subgroup, remove nodes from the graph
            partition.append(subgroup)
            remaining_nodes.difference_update(subgroup)
        #print(partition)
        return partition

    def fit(self, max_rounds):
        best_partition = None
        best_score = math.inf
        #find best partition through validation set
        for rounds in range(max_rounds):
            partition = self.greedy_clique_partition()
            score = 0
            for subgroup in partition:
                subgroup_array = np.asarray(list(subgroup))
                #print(subgroup_array)
                #sum validation weights of edges inside the group; ignore edges without data (nan weight)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    score = score + np.nansum(self.val_weights[subgroup_array[:,np.newaxis], subgroup_array])

            #print(score)
            if score <= best_score:
                best_partition = partition
                best_score = score
        print(best_score)
        return best_partition

    def visualize_pyvis(self, group_membership_list):
        net = Network()
        for g, members in enumerate(group_membership_list):
            for v in members:
                net.add_node(v, group = g, value =1)

        for v, neighbours in self.gdict.items():
            for n in neighbours:
                net.add_edge(v,n)

        net.toggle_physics(True)
        net.toggle_drag_nodes(True)
        net.show_buttons(filter_=['physics'])
        net.show('scm.html')

    def save(self, name, group_membership_list):
        G = nx.Graph()
        for g, members in enumerate(group_membership_list):
                G.add_nodes_from([ (v, {"group": g}) for v in members])

        for v, neighbours in self.gdict.items():
                G.add_edges_from([(v,n) for n in neighbours])

        nx.write_gexf(G, name + ".gexf")

        G_not = nx.complement(G)
        nx.write_gexf(G_not, name + "_complement.gexf")
        
        nx.draw_kamada_kawai(G, with_labels = True)
        plt.savefig("scm.png")

        nx.draw_kamada_kawai(G, with_labels = True)
        plt.savefig("scm_complement.png")


