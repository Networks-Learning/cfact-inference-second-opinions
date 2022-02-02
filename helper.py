#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

def overlap_coefficient(set1, set2):
    union_size = len(set1.intersection(set2))
    return union_size / len(set1)

def compare_groups(scm_trained, scm_real):
    real_group_list = scm_real.get_group_membership_list()
    trained_group_list = scm_trained.get_group_membership_list()

    #looks more reasonable:
    #scatter_factor = # groups the original group was split (no needed? -> size of the inner overlap_factor_list)
    #overlap_factor = % of the original group found in each training group
    overlap_factor = []
    overlap_per_trained = {}
    for group in real_group_list:
        trained_group_index = {scm_trained.get_group_index(ind) for ind in group}
        overlap_factor.append([ overlap_coefficient(group, trained_group_list[i]) for i in trained_group_index])
        for i in trained_group_index:
            overlap_per_trained[i] = overlap_per_trained.get(i, []) + [overlap_coefficient(group, trained_group_list[i])]
    scatter_factor = [len(l) for l in overlap_factor]
    print()
    print("Number of greedy groups each real group was split into")
    print(scatter_factor)
    print()
    print("Percentage of the real group in each greedy group")
    ratio_per_real = [ np.around(factor,decimals=2).tolist() for factor in overlap_factor]
    print(ratio_per_real)
    print()
    print("Percentage of each real group in a greedy group")
    ratio_per_greedy = [ np.around(factor,decimals=2).tolist() for g, factor in overlap_per_trained.items()]
    print(ratio_per_greedy)
    print()
    
    trained = [scm_trained.get_group_index(ind) for ind in range(scm_trained.n_experts)]
    real = [scm_real.get_group_index(ind) for ind in range(scm_trained.n_experts)]
    return adjusted_rand_score(trained, real)

def plot(s_list, N_list, real_mean, trained_mean, naive_mean, real_std, trained_std, naive_std):
    plt.title("Tradeoff #Training Samples and Sparsity s") 
    plt.xlabel("Number of training samples") 
    plt.ylabel("0/1-Score") 
    for i, s in enumerate(s_list):
        plt.errorbar(N_list, trained_mean[i], yerr=trained_std[i], fmt="-x", label="s = %.2f" % s) 
        plt.legend()
    plt.savefig("tradeoff.png")
    plt.show()
    
    for i, row in enumerate(trained_mean):
        plt.title("Comparison of Models for s = %.2f" % s_list[i]) 
        plt.xlabel("Number of training samples") 
        plt.ylabel("0/1-Score") 
        plt.ylim(0,1)
        plt.errorbar(N_list, trained_mean[i], yerr=trained_std[i], fmt="-o", label="Our SCM") 
        plt.errorbar(N_list, naive_mean[i], yerr=naive_std[i], fmt="-o", label="Naive SCM") 
        plt.errorbar(N_list, real_mean[i], yerr=real_std[i], fmt="-o", label="Real SCM") 
        plt.legend(loc="upper right")
        plt.savefig("comparison_%.2f.png"% s_list[i])
        plt.show()

def plot_group_comparison(s_list, N_list, mean, std):
    plt.title("Tradeoff #Training Samples and Sparsity s") 
    plt.xlabel("Number of training samples") 
    plt.ylabel("Adjusted Rand Index") 
    for i, s in enumerate(s_list):
        plt.errorbar(N_list, mean[i], yerr=std[i], fmt="-o", label="s = %.2f" % s) 
        plt.legend()
    plt.savefig("ARI_plot.png")
    plt.show()
 
def plot_rate_inedge(s_list, N_list, mean, std):
    plt.title("Tradeoff #Training Samples and Sparsity s") 
    plt.xlabel("Number of training samples") 
    plt.ylabel("Rate of edges inside real groups") 
    for i, s in enumerate(s_list):
        plt.errorbar(N_list, mean[i], yerr=std[i], fmt="-o", label="s = %.2f" % s) 
        plt.legend()
    plt.savefig("Inedge_plot.png")
    plt.show()
 
