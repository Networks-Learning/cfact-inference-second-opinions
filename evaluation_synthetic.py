#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import *

path="results_synthetic/"

def plot(s_list, N_list, real_mean, trained_mean, naive_mean, real_std, trained_std, naive_std):
    cmap = get_cmap(5)
    w,h = get_fig_dim(width=487,fraction=0.7)
    fig, ax = plt.subplots(figsize=(w,h))
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("0/1-Score")
    ax.set_ylim(0.2,1)
    ax.errorbar(N_list, naive_mean[-1], yerr=naive_std[-1], fmt="-o", c=cmap(0), label="Marginal Distr. M(H)")
    ax.errorbar(N_list, real_mean[-1], yerr=real_std[-1], fmt="-o", c=cmap(7),label="True SI-SCM M*")
    ax.errorbar(N_list, trained_mean[-5], yerr=trained_std[-5], fmt="-o", c=cmap(6), label="Gumbel-Max SI-SCM for s=%.2f"% s_list[-5])
    ax.legend(loc="upper right")
    plt.savefig(path+"comparison_1.pdf")
    plt.show()

    fig, ax = plt.subplots(figsize=(w,h))
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("0/1-Score")
    ax.set_ylim(0.2,1)
    ax.errorbar(N_list, np.nanmean(naive_mean, axis=0), yerr=np.nanmean(naive_std, axis=0), fmt="-o", c=cmap(0), label=r"Marginal Distr. $P(Y_h \ \mid\ X=x)$")
    ax.errorbar(N_list, np.nanmean(real_mean, axis=0), yerr=np.nanmean(real_std, axis=0), fmt="-o", c=cmap(7),label=r"True SI-SCM $\mathcal{M}(\Psi)$")
    ax.errorbar(N_list, trained_mean[-4], yerr=trained_std[-4], fmt="-o", c=cmap(5), label="SI-SCM $\mathcal{M}(\hat{\Psi})$ for s=%.2f"% s_list[-4])
    ax.errorbar(N_list, trained_mean[-3], yerr=trained_std[-3], fmt="-o", c=cmap(4),label="SI-SCM $\mathcal{M}(\hat{\Psi})$ for s=%.2f"% s_list[-3])
    ax.errorbar(N_list, trained_mean[-2], yerr=trained_std[-2], fmt="-o", c=cmap(3), label="SI-SCM $\mathcal{M}(\hat{\Psi})$ for s=%.2f"% s_list[-2])
    ax.errorbar(N_list, trained_mean[-1], yerr=trained_std[-1], fmt="-o", c=cmap(2),label="SI-SCM $\mathcal{M}(\hat{\Psi})$ for s=%.2f"% s_list[-1])
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(path+"comparison_2.pdf")
    plt.show()

def plot_group_comparison(s_list, N_list, mean_ari, std_ari):
    n= s_list.shape[0]
    cmap = get_cmap(n)
    w,h = get_fig_dim(width=487,fraction=0.7)
    fig, ax = plt.subplots(figsize=(w,h))
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Adjusted Random Index (ARI)")
    for i, s in enumerate(s_list):
        if i in [0]: continue
        ax.errorbar(N_list, mean_ari[i], yerr=std_ari[i], fmt="-o", c=cmap(n-i+1), label="s = %.2f" % s)
    ax.legend()
    fig.tight_layout()
    plt.savefig(path+"ARI_plot.pdf")
    plt.show()

def plot_rate_inedge(s_list, N_list, mean_rate, std_rate):
    n= s_list.shape[0]
    cmap = get_cmap(n)
    w,h = get_fig_dim(width=487,fraction=0.7)
    fig, ax = plt.subplots(figsize=(w,h))
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Edge Ratio")
    for i, s in enumerate(s_list):
        if i in [0]: continue
        ax.errorbar(N_list, mean_rate[i], yerr=std_rate[i], c=cmap(n-i+1), fmt="--o", label="s = %.2f" % s)
    ax.legend()
    fig.tight_layout()
    plt.savefig(path+"Inedge_plot.pdf")
    plt.show()

def get_cmap(n, name='gist_earth'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n+5)

#def plot_rate_ARI(s_list, N_list, mean_rate, mean_ari, std_rate, std_ari):
#    n= s_list.shape[0]
#    cmap = get_cmap(n)
#
#    fig, ax = plt.subplots()
#    ax.set_xlabel("Number of training samples")
#    #plt.ylabel("Rate of edges inside real groups")
#    for i, s in enumerate(s_list):
#        if i in [1,3,4]: continue
#        ax.errorbar(N_list, mean_ari[i], yerr=std_ari[i], fmt="-o", c=cmap(n-i), label="s = %.2f" % s)
#
#    ax2 = ax.twinx()
#    for i, s in enumerate(s_list):
#        if i in [1,3,4]: continue
#        ax.errorbar(N_list, mean_rate[i], yerr=std_rate[i], c=cmap(n-i), fmt="--o")
#    
#    ax2.plot([],[], ls="-",c='black', label='ARI values')
#    ax2.plot([],[], ls="--",c='black', label='Edge Ratio')
#    
#    ax.legend(loc=4)
#    ax2.legend(loc=1)
#    plt.savefig("Rate_ARI_plot_1.pdf")
#    plt.show()
#
#    fig, ax = plt.subplots()
#    ax.set_xlabel("Number of training samples")
#    #plt.ylabel("Rate of edges inside real groups")
#    for i, s in enumerate(s_list):
#        if i in [0,2]: continue
#        ax.errorbar(N_list, mean_rate[i], yerr=std_rate[i], c=cmap(n-i), fmt="--o", label="s = %.2f" % s)
#
#    ax2 = ax.twinx()
#    for i, s in enumerate(s_list):
#        if i in [0,2]: continue
#        ax.errorbar(N_list, mean_ari[i], yerr=std_ari[i], fmt="-o", c=cmap(n-i))
#    
#    ax2.plot([],[], ls="-",c='black', label='ARI values')
#    ax2.plot([],[], ls="--",c='black', label='Edge Rate')
#    
#    ax.legend(loc=4)
#    ax2.legend(loc=1)
#    plt.savefig(path+"Rate_ARI_plot_2.pdf")
#    plt.show()



def main():
    latexify()
    mean_trained_df = pd.read_csv(path+"mean_trained.csv", header =0, index_col = 0)
    s_list = mean_trained_df.index.to_numpy(dtype=float)
    N_list = mean_trained_df.columns.to_numpy(dtype=int)
    print("Reading Data...")
    mean_trained = mean_trained_df.to_numpy(na_value=np.nan)
    mean_naive = pd.read_csv(path+"mean_naive.csv", header =0, index_col = 0).to_numpy(na_value=np.nan)
    mean_real = pd.read_csv(path+"mean_real.csv", header =0, index_col = 0).to_numpy(na_value=np.nan)
    mean_groups = pd.read_csv(path+"mean_groups.csv", header =0, index_col = 0).to_numpy(na_value=np.nan)
    mean_inedge = pd.read_csv(path+"mean_inedge.csv", header =0, index_col = 0).to_numpy(na_value=np.nan)
    
    std_trained = pd.read_csv(path+"std_trained.csv", header =0, index_col = 0).to_numpy(na_value=np.nan)
    std_naive = pd.read_csv(path+"std_naive.csv", header =0, index_col = 0).to_numpy(na_value=np.nan)
    std_real = pd.read_csv(path+"std_real.csv", header =0, index_col = 0).to_numpy(na_value=np.nan)
    std_groups = pd.read_csv(path+"std_groups.csv", header =0, index_col = 0).to_numpy(na_value=np.nan)
    std_inedge = pd.read_csv(path+"std_inedge.csv", header =0, index_col = 0).to_numpy(na_value=np.nan)
    
    print("Generating Plots...")
    plot_rate_inedge(s_list, N_list, mean_inedge, std_inedge)
    plot_group_comparison(s_list, N_list, mean_groups, std_groups)
    #plot_rate_ARI(s_list, N_list,mean_inedge, mean_groups, std_inedge,std_groups)
    plot(s_list, N_list, mean_real, mean_trained, mean_naive, std_real, std_trained, std_naive)

if __name__ == "__main__":
    main()


