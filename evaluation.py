#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

def eval_plot_per_expert(eval_matrix, model_name, n_experts):
    print("Accuracy per expert...",model_name)
    acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,2]==exp) for exp in range(n_experts)])
    _ = plt.hist(acc, bins=np.arange(0, 1.1, 0.1))
    plt.title("Histogram of Accuracy per Expert: "+ model_name)
    plt.savefig("hist_E_"+model_name)
    plt.show()

def eval_plot_per_expert_group(eval_matrix, model_name, n_experts):
    print("Accuracy per expert...",model_name)
    print(np.sum(eval_matrix[:,6]==1))
    acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,2]==exp) & (eval_matrix[:,6]==1)) for exp in range(n_experts)])
    _ = plt.hist(acc, bins=np.arange(0, 1.1, 0.1))
    plt.title("Histogram of Accuracy per Expert & same group: "+ model_name)
    plt.savefig("hist_GE_"+model_name)
    plt.show()

def eval_plot_per_expert_group_diff(eval_matrix, model_name, n_experts):
    print("Accuracy per expert...",model_name)
    print(np.sum((eval_matrix[:,6]==1) & (eval_matrix[:,4]!=eval_matrix[:,3])))
    print(np.sum((eval_matrix[:,6]==1) & (eval_matrix[:,5]!=eval_matrix[:,3])))
    #print([np.sum((eval_matrix[:,0]==d) & (eval_matrix[:,6]==1) & (eval_matrix[:,4]!=eval_matrix[:,3])) for d in range(int(np.max(eval_matrix[:,0])))])
    acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,2]==exp) & (eval_matrix[:,6]==1) & (eval_matrix[:,4]!=eval_matrix[:,3])) for exp in range(n_experts)])
    _ = plt.hist(acc, bins=np.arange(0, 1.1, 0.1))
    plt.title("Histogram of Accuracy per Expert & same group & different predictions: "+ model_name)
    plt.savefig("hist_GED_"+model_name)
    plt.show()

def main():
    siscm_results = pd.read_csv('data/evaluation_results_trained.csv').to_numpy()
    naive_results = pd.read_csv('data/evaluation_results_naive.csv').to_numpy()
    logreg_baseline_results = pd.read_csv('data/evaluation_results_logreg_baseline.csv').to_numpy()
    nb_baseline_results = pd.read_csv('data/evaluation_results_nb_baseline.csv').to_numpy()
    n_experts = int(np.max(siscm_results[:,2]))
    eval_plot_per_expert(siscm_results, "Trained SISCM", n_experts)
    eval_plot_per_expert(naive_results, "Naive SISCM", n_experts)
    eval_plot_per_expert(logreg_baseline_results, "LogReg+Obs Baseline", n_experts)
    eval_plot_per_expert(nb_baseline_results, "NB+Obs Baseline", n_experts)

    eval_plot_per_expert_group(siscm_results, "Trained SISCM", n_experts)
    eval_plot_per_expert_group(naive_results, "Naive SISCM", n_experts)
    eval_plot_per_expert_group(logreg_baseline_results, "LogReg+Obs Baseline", n_experts)
    eval_plot_per_expert_group(nb_baseline_results, "NB+Obs Baseline", n_experts)

    eval_plot_per_expert_group_diff(siscm_results, "Trained SISCM", n_experts)
    eval_plot_per_expert_group_diff(naive_results, "Naive SISCM", n_experts)
    eval_plot_per_expert_group_diff(logreg_baseline_results, "LogReg+Obs Baseline", n_experts)
    eval_plot_per_expert_group_diff(nb_baseline_results, "NB+Obs Baseline", n_experts)



if __name__ == "__main__":
    main()
