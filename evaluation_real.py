#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from helper import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable 

np.set_printoptions(precision=3)
path = "results_real/"

def plot_confusion_matrix(eval_matrix, model_name):
    w,h = get_fig_dim(width=487,fraction=0.7)
    fig, ax = plt.subplots(figsize=(w,h))
    print("Confusion Matrix...",model_name)
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog","frog","horse","ship","truck"]
    #cm = confusion_matrix( eval_matrix[:,4], eval_matrix[:,5], labels= np.arange(10, dtype=int), normalize='true')
    ConfusionMatrixDisplay.from_predictions(eval_matrix[:,4], eval_matrix[:,5],normalize='true', display_labels=labels, values_format='.2f', xticks_rotation='vertical', cmap='YlGnBu', ax=ax, colorbar=False)
    ax.set_xlabel("Model's Prediction of Expert's Label")
    ax.set_ylabel("Expert's Label Prediction")
    fig.tight_layout()
    plt.savefig(path+"CM_"+model_name+".pdf")
    plt.show()


def plot_expert_acc(trained, baseline, baseline_name, n_experts,ax):
    acc_trained = np.array([np.mean(trained[:,4]==trained[:,5], where= trained[:,2]==exp) for exp in range(n_experts)])
    acc_baseline = np.array([np.mean(baseline[:,4]==baseline[:,5], where= baseline[:,2]==exp) for exp in range(n_experts)])
    print("scenario 1 - #experts not displayed: ", np.arange(n_experts,dtype=int)[np.isnan(acc_trained)])
    print("scenario 1 - #experts Gumbel-Max SI-SCM is more accurate for than ",baseline_name,": ", np.sum(acc_trained>acc_baseline))
    H, x_edges, y_edges = np.histogram2d(acc_trained, acc_baseline, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    #plt.title(" Accuracy per Expert SI-SCM vs. "+ baseline_name)
    X, Y = np.meshgrid(x_edges, y_edges)
    im = ax.pcolormesh(X, Y, H, cmap='YlGnBu', vmin=0, vmax=6)
    ax.axline((1,1),slope=1, ls="--", color="red")
    ax.plot(np.nanmean(acc_baseline), np.nanmean(acc_trained), 'rx', markersize=8)
    ax.set_xlabel('Accuracy '+baseline_name)
    ax.set_ylabel('Accuracy G.-M. SI-SCM')
    return im

def plot_expert_acc_same_group(trained, baseline, baseline_name, n_experts, ax):
    acc_trained = np.array([np.mean(trained[:,4]==trained[:,5], where= (trained[:,2]==exp) & (trained[:,6]==1)) for exp in range(n_experts)])
    acc_baseline = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==1)) for exp in range(n_experts)])
    print("scenario 2 - #experts not displayed: ", np.arange(n_experts,dtype=int)[np.isnan(acc_trained)])
    print("scenario 2 - #experts Gumbel-Max SI-SCM is more accurate for than ",baseline_name,": ", np.sum(acc_trained>acc_baseline))
    H, x_edges, y_edges = np.histogram2d(acc_trained, acc_baseline, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    X, Y = np.meshgrid(x_edges, y_edges)
    im = ax.pcolormesh(X, Y, H, cmap='YlGnBu', vmin=0, vmax=6)
    ax.axline((1,1),slope=1, ls="--", color="red") 
    ax.plot(np.nanmean(acc_baseline), np.nanmean(acc_trained), 'rx', markersize=8)
    ax.set_xlabel('Accuracy '+baseline_name)
    ax.set_ylabel('Accuracy G.-M. SI-SCM')
    return im

def plot_expert_acc_diffgroup(trained, baseline, baseline_name, n_experts):
    acc_trained = np.array([np.mean(trained[:,4]==trained[:,5], where= (trained[:,2]==exp) & (trained[:,6]==0)) for exp in range(n_experts)])
    acc_baseline = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==0)) for exp in range(n_experts)])
    H, x_edges, y_edges = np.histogram2d(acc_trained, acc_baseline, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H, cmap='YlGnBu')
    plt.axline((1,1),slope=1, ls="--", color="red") 
    plt.plot(np.nanmean(acc_baseline), np.nanmean(acc_trained), 'rx', markersize=8)
    plt.xlabel('Accuracy '+baseline_name )
    plt.ylabel('Accuracy Gumbel-Max SI-SCM')

    plt.colorbar()
    plt.savefig(path+"hist2d_NGE_"+baseline_name)
    plt.show()

def plot_expert_acc_same_vs_diff_group( baseline, baseline_name, n_experts,ax):
    acc_group = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==1)) for exp in range(n_experts)])
    acc_not_group = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==0)) for exp in range(n_experts)])
    print("scenario 2 vs.3 - #experts ", baseline_name," more accurate in 2 than 3: ", np.sum(acc_group>acc_not_group))
    H, x_edges, y_edges = np.histogram2d(acc_group, acc_not_group, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    X, Y = np.meshgrid(x_edges, y_edges)
    im =ax.pcolormesh(X, Y, H, cmap='YlGnBu', vmin=0, vmax=7)
    ax.axline((1,1),slope=1, ls="--", color="red")
    ax.plot(np.nanmean(acc_not_group), np.nanmean(acc_group), 'rx', markersize=8)
    ax.set_ylabel(r"Accuracy for h,h' $\in \psi$")
    ax.set_xlabel(r"Accuracy for h,h' $\in \mathcal{H}$")
    return im

def print_overall_acc(trained, untrained, nb):
    trained_acc = np.mean(trained[:,4]==trained[:,5])
    trained_acc_g = np.mean(trained[:,4]==trained[:,5], where= trained[:,6]==1)
    trained_acc_ng = np.mean(trained[:,4]==trained[:,5], where= trained[:,6]==0)

    untrained_acc = np.mean(untrained[:,4]==untrained[:,5])
    untrained_acc_g = np.mean(untrained[:,4]==untrained[:,5], where= untrained[:,6]==1)
    untrained_acc_ng = np.mean(untrained[:,4]==untrained[:,5], where= untrained[:,6]==0)

    nb_acc = np.mean(nb[:,4]==nb[:,5])
    nb_acc_g = np.mean(nb[:,4]==nb[:,5], where= nb[:,6]==1)
    nb_acc_ng = np.mean(nb[:,4]==nb[:,5], where= nb[:,6]==0)
    
    print("Model\t: GM-SI-SCM \t GNB \t GNB+CNB")
    print("Acc scenario 1\t:", trained_acc, untrained_acc, nb_acc)
    print("Acc scenario 2\t:", trained_acc_g, untrained_acc_g, nb_acc_g)
    print("Acc scnario 3 \t:", trained_acc_ng, untrained_acc_ng, nb_acc_ng)

def main():
    latexify(font_size=10)
    #read result files
    siscm_psi_results = pd.read_csv(path+"evaluation_results_SISCM_M(Psi).csv").to_numpy()
    siscm_H_results = pd.read_csv(path+"evaluation_results_SISCM_M(H).csv").to_numpy()
    nb_baseline_results = pd.read_csv(path+"evaluation_results_nb_baseline.csv").to_numpy()
    #gnb_results = pd.read_csv(path+"evaluation_results_proba_models.csv").to_numpy()
 
    #Meta-data about the real experiment   
    n_experts = int(np.max(siscm_psi_results[:,2])+1)
    n_data_test = int(np.max(siscm_psi_results[:,0])+1)
    disagreement = [np.mean((siscm_H_results[:,4]!=siscm_H_results[:,3]), where=siscm_H_results[:,2]==exp) for exp in range(n_experts)]
    print("# data test: ",n_data_test)
    print("# experts: ",n_experts)
    print("Disagreement ratio: ",np.nanmean(disagreement))
    n_pred = siscm_psi_results.shape[0]
    n_pred_group = int(np.sum(siscm_psi_results[:,6]==1))
    n_pred_notgroup = int(np.sum(siscm_psi_results[:,6]==0))
    print("#Pred: ", n_pred)
    print("#Pred Group: ", n_pred_group)
    print("#Pred not Group: ", n_pred_notgroup)

 ##################
    #print overall accuracy results of the three models
    print_overall_acc(siscm_psi_results, siscm_H_results, nb_baseline_results)

 ##################
    #plot group sizes of SI-SCM M(Psi) as vertical barplot
    groups = pd.read_csv(path+"SI-SCM_groups.csv",header=None).to_numpy()
    group_sizes = np.sum(~np.isnan(groups), axis=1).T

    w,h = get_fig_dim(width=487,fraction=0.7)
    fig, axes = plt.subplots(figsize=(w,h))
    axes.set_ylabel("Mutually Similar Groups")
    axes.set_xlabel("Number of Experts in each Group")
    # get rid of the frame
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    # remove all the ticks and directly label each bar with respective value
    axes.xaxis.set_ticks_position('none')
    axes.xaxis.set_ticks([])
    axes.yaxis.set_ticks(np.arange(group_sizes.shape[0])+1)

    axes.barh(y=np.arange(group_sizes.shape[0])+1, width=group_sizes)
    axes.bar_label(axes.containers[0], padding=3)
    fig.tight_layout()
    plt.savefig(path+"groups_hist.pdf")
    plt.show()
###################
    #plot confusion matrices
    plot_confusion_matrix(siscm_psi_results, "Gumbel-Max SI-SCM")
    plot_confusion_matrix(siscm_H_results, "GNB")
    plot_confusion_matrix( nb_baseline_results, "GNB+CNB")

###################
    #plot per expert accuracy
    w,h = get_fig_dim(width=487,fraction=0.4)
    fig, axes = plt.subplots(figsize=(w,h))
    im =plot_expert_acc(siscm_psi_results, siscm_H_results, "GNB", n_experts,axes)
    axes.set(aspect=1)
    #plt.colorbar(im, location='right',shrink=0.7)
    fig.tight_layout()
    plt.savefig(path+"compare_untrained_sc1.pdf")
    plt.show()
 
    fig, axes = plt.subplots(figsize=(w,h))
    im = plot_expert_acc(siscm_psi_results, nb_baseline_results, "GNB+CNB", n_experts, axes)
    axes.set(aspect=1)
    #plt.colorbar(im, location='right',shrink=0.7)
    fig.tight_layout()
    plt.savefig(path+"compare_naivebayes_sc1.pdf")
    plt.show()

    w, _ = get_fig_dim(width=487,fraction=0.48)
    fig, axes = plt.subplots(figsize=(w,h))
    axes.set(aspect=1)
    #plt.colorbar(im, location='right',shrink=0.7)
    divider = make_axes_locatable(axes)
    ax_cb = divider.new_horizontal(size='5%', pad='10%')
    fig = axes.get_figure()
    fig.add_axes(ax_cb)
    im =plot_expert_acc_same_group(siscm_psi_results, siscm_H_results, "GNB", n_experts,axes)
    matplotlib.colorbar.ColorbarBase(ax_cb, cmap='YlGnBu', norm=matplotlib.colors.Normalize(vmin=0, vmax=6), orientation='vertical')#, ticks=[0,1,2,3,4])
    fig.tight_layout()
    plt.savefig(path+"compare_untrained_sc2.pdf")
    plt.show()
    
    fig, axes = plt.subplots(figsize=(w,h))
    axes.set(aspect=1)
    #plt.colorbar(im, location='right',shrink=0.7)
    divider = make_axes_locatable(axes)
    ax_cb = divider.new_horizontal(size='5%', pad='10%')
    fig = axes.get_figure()
    fig.add_axes(ax_cb)
    im = plot_expert_acc_same_group(siscm_psi_results, nb_baseline_results, "GNB+CNB", n_experts,axes)
    matplotlib.colorbar.ColorbarBase(ax_cb, cmap='YlGnBu', norm=matplotlib.colors.Normalize(vmin=0, vmax=6), orientation='vertical')#, ticks=[0,1,2,3,4])
    fig.tight_layout()
    plt.savefig(path+"compare_naivebayes_sc2.pdf")
    plt.show()
########### 
    #plot GNB+CNB baseline results in groups vs across groups
    w,h = get_fig_dim(width=487,fraction=0.5)
    fig, axes = plt.subplots(figsize=(w,h))
    axes.set(aspect=1)
    divider = make_axes_locatable(axes)
    ax_cb = divider.new_horizontal(size='5%', pad='10%')
    fig = axes.get_figure()
    fig.add_axes(ax_cb)
    im = plot_expert_acc_same_vs_diff_group( nb_baseline_results, "GNB+CNB", n_experts, axes)
    #plt.colorbar(im, location='right',shrink=0.7)
    matplotlib.colorbar.ColorbarBase(ax_cb, cmap='YlGnBu', norm=matplotlib.colors.Normalize(vmin=0, vmax=7), orientation='vertical')
    fig.tight_layout()
    plt.savefig(path+"nb_baseline.pdf")
    plt.show()
       
  ###################


if __name__ == "__main__":
    main()
