#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

def eval_plot_per_expert(eval_matrix, model_name, n_experts):
    print("Accuracy per expert...",model_name)
    acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,2]==exp) for exp in range(n_experts)])
    _ = plt.hist(acc, bins=np.arange(0, 1.1, 0.025))
    plt.title("Histogram of Accuracy per Expert: "+ model_name)
    plt.savefig("hist_E_"+model_name)
    plt.show()

def eval_plot_per_expert_group(eval_matrix, model_name, n_experts):
    print("Accuracy per expert...",model_name)
    print(np.sum(eval_matrix[:,6]==1))
    acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,2]==exp) & (eval_matrix[:,6]==1)) for exp in range(n_experts)])
    print("Mean: ", np.nanmean(acc))
    _ = plt.hist(acc, bins=np.arange(0, 1.1, 0.025))
    plt.title("Histogram of Accuracy per Expert & same group: "+ model_name)
    plt.savefig("hist_GE_"+model_name)
    plt.show()

def eval_plot_per_expert_nongroup(eval_matrix, model_name, n_experts):
    print("Accuracy per expert...",model_name)
    print(np.sum(eval_matrix[:,6]==1))
    acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,2]==exp) & (eval_matrix[:,6]==0)) for exp in range(n_experts)])
    print("Mean: ", np.nanmean(acc))
    _ = plt.hist(acc, bins=np.arange(0, 1.1, 0.025))
    plt.title("Histogram of Accuracy per Expert & diff group: "+ model_name)
    plt.savefig("hist_NGE_"+model_name)
    plt.show()

def eval_plot_per_expert_group_diff(eval_matrix, model_name, n_experts):
    print("Accuracy per expert...",model_name)
    print(np.sum((eval_matrix[:,6]==1) & (eval_matrix[:,4]!=eval_matrix[:,3])))
    print(np.sum((eval_matrix[:,6]==1) & (eval_matrix[:,5]!=eval_matrix[:,3])))
    #print([np.sum((eval_matrix[:,0]==d) & (eval_matrix[:,6]==1) & (eval_matrix[:,4]!=eval_matrix[:,3])) for d in range(int(np.max(eval_matrix[:,0])))])
    acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,2]==exp) & (eval_matrix[:,6]==1) & (eval_matrix[:,4]!=eval_matrix[:,3])) for exp in range(n_experts)])
    _ = plt.hist(acc, bins=np.arange(0, 1.1, 0.025))
    plt.title("Histogram of Accuracy per Expert & same group & different predictions: "+ model_name)
    plt.savefig("hist_GED_"+model_name)
    plt.show()

def plot_diff_acc(trained, baseline, baseline_name, n_experts):
    acc_trained = np.array([np.mean(trained[:,4]==trained[:,5], where= trained[:,2]==exp) for exp in range(n_experts)])
    acc_baseline = np.array([np.mean(baseline[:,4]==baseline[:,5], where= baseline[:,2]==exp) for exp in range(n_experts)])
    diff_acc = acc_trained - acc_baseline
    _ = plt.hist(diff_acc, bins=np.arange(-0.5, 0.6, 0.025))
    plt.title("Histogram of Diff. Accuracy per Expert vs. "+ baseline_name)
    plt.savefig("hist_Diff_E_"+baseline_name)
    plt.show()
 
def plot_diff_acc_2D(trained, baseline, baseline_name, n_experts):
    acc_trained = np.array([np.mean(trained[:,4]==trained[:,5], where= trained[:,2]==exp) for exp in range(n_experts)])
    acc_baseline = np.array([np.mean(baseline[:,4]==baseline[:,5], where= baseline[:,2]==exp) for exp in range(n_experts)])
    H, x_edges, y_edges = np.histogram2d(acc_trained, acc_baseline, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    plt.title(" Accuracy per Expert SI-SCM vs. "+ baseline_name)
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H)
    plt.axline((1,1),slope=1, ls="--", color="red")
    plt.colorbar()
    plt.savefig("hist2d_E_"+baseline_name)
    plt.show()

def plot_diff_acc_group(trained, baseline, baseline_name, n_experts):
    acc_trained = np.array([np.mean(trained[:,4]==trained[:,5], where= (trained[:,2]==exp) & (trained[:,6]==1)) for exp in range(n_experts)])
    acc_baseline = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==1)) for exp in range(n_experts)])
    diff_acc = acc_trained - acc_baseline
    print(diff_acc)
    _ = plt.hist(diff_acc, bins=np.arange(-0.5, 0.5, 0.025))
    plt.title("Histogram of Diff. Accuracy per Expert vs. "+ baseline_name)
    plt.savefig("hist_Diff_GE_"+baseline_name)
    plt.show()   

def plot_diff_acc_group_2D(trained, baseline, baseline_name, n_experts):
    acc_trained = np.array([np.mean(trained[:,4]==trained[:,5], where= (trained[:,2]==exp) & (trained[:,6]==1)) for exp in range(n_experts)])
    acc_baseline = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==1)) for exp in range(n_experts)])
    H, x_edges, y_edges = np.histogram2d(acc_trained, acc_baseline, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    plt.title("Accuracy per Expert (group obs.) SI-SCM vs. "+ baseline_name)
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H)
    plt.axline((1,1),slope=1, ls="--", color="red")
    plt.colorbar()
    plt.savefig("hist2d_GE_"+baseline_name)
    plt.show()

def plot_diff_acc_nongroup_2D(trained, baseline, baseline_name, n_experts):
    acc_trained = np.array([np.mean(trained[:,4]==trained[:,5], where= (trained[:,2]==exp) & (trained[:,6]==0)) for exp in range(n_experts)])
    acc_baseline = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==0)) for exp in range(n_experts)])
    H, x_edges, y_edges = np.histogram2d(acc_trained, acc_baseline, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    plt.title("Accuracy per Expert (non group obs.) SI-SCM vs. "+ baseline_name)
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H)
    plt.axline((1,1),slope=1, ls="--", color="red")
    plt.colorbar()
    plt.savefig("hist2d_NGE_"+baseline_name)
    plt.show()

def plot_acc_gng_2D( baseline, baseline_name, n_experts):
    acc_group = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==1)) for exp in range(n_experts)])
    acc_not_group = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==0)) for exp in range(n_experts)])
    H, x_edges, y_edges = np.histogram2d(acc_group, acc_not_group, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    plt.title("Accuracy per Expert group vs. no group obs.: "+ baseline_name)
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H)
    plt.axline((1,1),slope=1, ls="--", color="red")
    plt.colorbar()
    plt.savefig("hist2d_EGNG_"+baseline_name)
    plt.show()

def plot_diff_gng( baseline, baseline_name, n_experts):
    acc_group = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==1)) for exp in range(n_experts)])
    acc_not_group = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==0)) for exp in range(n_experts)])
    diff_acc = acc_group - acc_not_group
    _ = plt.hist(diff_acc, bins=np.arange(-0.5, 0.5, 0.025))
    plt.title("Diff. Accuracy per Expert group vs no group obs.: "+ baseline_name)
    plt.savefig("hist_Diff_EGNG_"+baseline_name)
    plt.show()   

def main():
    #path = "data/"
    #path = "data/large_results/"
    path = "data/clique_results/"
    #path = "data/param(0,50,10,100)/"
    #siscm_results = pd.read_csv('data/clique_results/evaluation_results_trained.csv').to_numpy()
    #naive_results = pd.read_csv('data/clique_results/evaluation_results_naive.csv').to_numpy()
    #logreg_baseline_results = pd.read_csv('data/clique_results/evaluation_results_logreg_baseline.csv').to_numpy()
    #nb_baseline_results = pd.read_csv('data/clique_results/evaluation_results_nb_baseline.csv').to_numpy()
    siscm_results = pd.read_csv(path+"evaluation_results_trained.csv").to_numpy()
    naive_results = pd.read_csv(path+"evaluation_results_naive.csv").to_numpy()
    logreg_baseline_results = pd.read_csv(path+"evaluation_results_logreg_baseline.csv").to_numpy()
    nb_baseline_results = pd.read_csv(path+"evaluation_results_nb_baseline.csv").to_numpy()
 
    n_experts = int(np.max(siscm_results[:,2]))

    plot_diff_acc_2D(siscm_results, naive_results, "Naive", n_experts)
    plot_diff_acc_2D(siscm_results, logreg_baseline_results, "LogReg", n_experts)
    plot_diff_acc_2D(siscm_results, nb_baseline_results, "G+CatNB", n_experts)

    plot_diff_acc_group_2D(siscm_results, naive_results, "Naive", n_experts)
    plot_diff_acc_nongroup_2D(siscm_results, naive_results, "Naive", n_experts)
    plot_diff_acc_group_2D(siscm_results, logreg_baseline_results, "LogReg", n_experts)
    plot_diff_acc_group_2D(siscm_results, nb_baseline_results, "G+CatNB", n_experts)

    plot_acc_gng_2D(siscm_results, "Trained SI-SCM", n_experts)
    plot_acc_gng_2D( logreg_baseline_results, "LogReg", n_experts)
    plot_acc_gng_2D( nb_baseline_results, "G+CatNB", n_experts)

    plot_diff_gng(siscm_results, "Trained SI-SCM", n_experts)
    plot_diff_gng( logreg_baseline_results, "LogReg", n_experts)
    plot_diff_gng( nb_baseline_results, "G+CatNB", n_experts)



    plot_diff_acc(siscm_results, naive_results, "Naive", n_experts)
    plot_diff_acc(siscm_results, logreg_baseline_results, "LogReg", n_experts)
    plot_diff_acc(siscm_results, nb_baseline_results, "G+CatNB", n_experts)

    plot_diff_acc_group(siscm_results, naive_results, "Naive", n_experts)
    plot_diff_acc_group(siscm_results, logreg_baseline_results, "LogReg", n_experts)
    plot_diff_acc_group(siscm_results, nb_baseline_results, "G+CatNB", n_experts)

    eval_plot_per_expert(siscm_results, "Trained SISCM", n_experts)
    #eval_plot_per_expert(naive_results, "Naive SISCM", n_experts)
    #eval_plot_per_expert(logreg_baseline_results, "LogReg Baseline", n_experts)
    #eval_plot_per_expert(nb_baseline_results, "G+CatNB Baseline", n_experts)

    eval_plot_per_expert_group(siscm_results, "Trained SISCM", n_experts)
    #eval_plot_per_expert_group(naive_results, "Naive SISCM", n_experts)
    eval_plot_per_expert_group(logreg_baseline_results, "LogReg+Obs Baseline", n_experts)
    eval_plot_per_expert_group(nb_baseline_results, "G+CatNB Baseline", n_experts)
    
    eval_plot_per_expert_nongroup(siscm_results, "Trained SISCM", n_experts)
    eval_plot_per_expert_nongroup(logreg_baseline_results, "LogReg SISCM", n_experts)
    eval_plot_per_expert_nongroup(nb_baseline_results, "G+CatNB SISCM", n_experts)

    eval_plot_per_expert_group_diff(siscm_results, "Trained SISCM", n_experts)
    #eval_plot_per_expert_group_diff(naive_results, "Naive SISCM", n_experts)
    #eval_plot_per_expert_group_diff(logreg_baseline_results, "LogReg+Obs Baseline", n_experts)
    #eval_plot_per_expert_group_diff(nb_baseline_results, "G+CatNB Baseline", n_experts)



if __name__ == "__main__":
    main()
