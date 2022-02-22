#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

np.set_printoptions(precision=3)
#path = "data/"
#path = "data/results_seed3993/"
path = "data/results_seed42_140/"

def get_fig_dim(width=487, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (1 + 5**.5) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / golden_ratio

    fig_dim = (fig_height_in, golden_ratio)

    return fig_dim


def latexify(font_serif='Computer Modern', mathtext_font='cm', font_size=10, small_font_size=None, usetex=True):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.
    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    if small_font_size is None:
      small_font_size = font_size
    
    params = {'backend': 'ps',
              'text.latex.preamble': '\\usepackage{gensymb} \\usepackage{bm}',
              # fontsize for x and y labels (was 10)
            #   'axes.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
            #   'axes.titlesize': font_scale * 10 if largeFonts else font_scale * 7,
            #   'font.size': font_scale * 10 if largeFonts else font_scale * 7,  # was 10
            #   'legend.fontsize': font_scale * 10 if largeFonts else font_scale * 7,  # was 10
            #   'xtick.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
            #   'ytick.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'font.size': font_size,  # was 10
              'legend.fontsize': small_font_size,  # was 10
              'legend.title_fontsize': small_font_size,
              'xtick.labelsize': small_font_size,
              'ytick.labelsize': small_font_size,
              'text.usetex': usetex,
            #   'figure.figsize': [fig_width, fig_height],
              'font.family' : 'serif',
              'font.serif' : font_serif,
              'mathtext.fontset' : mathtext_font
            #   'xtick.minor.size': 0.5,
            #   'xtick.major.pad': 1.5,
            #   'xtick.major.size': 1,
            #   'ytick.minor.size': 0.5,
            #   'ytick.major.pad': 1.5,
            #   'ytick.major.size': 1,
            # #   'lines.linewidth': 1.5,
            # 'lines.linewidth': 1,
            # #   'lines.markersize': 0.1,
            #   'lines.markersize': 8.0,
            #   'hatch.linewidth': 0.5
              }

    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)

def plot_confusion_matrix(eval_matrix, model_name):
    print("Confusion Matrix...",model_name)
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog","frog","horse","ship","truck"]
    #cm = confusion_matrix( eval_matrix[:,4], eval_matrix[:,5], labels= np.arange(10, dtype=int), normalize='true')
    ConfusionMatrixDisplay.from_predictions(eval_matrix[:,4], eval_matrix[:,5],normalize='true', display_labels=labels, values_format='.2f', xticks_rotation='vertical', cmap='YlGnBu')
    plt.title("Confusion Matrix: "+ model_name)
    plt.savefig(path+"CM_"+model_name)
    plt.show()


def plot_diff_acc_2D(trained, baseline, baseline_name, n_experts):
    acc_trained = np.array([np.mean(trained[:,4]==trained[:,5], where= trained[:,2]==exp) for exp in range(n_experts)])
    acc_baseline = np.array([np.mean(baseline[:,4]==baseline[:,5], where= baseline[:,2]==exp) for exp in range(n_experts)])
    print("scenario 1 - #experts trained SI-SCM is more accurate for than ",baseline_name,": ", np.sum(acc_trained>acc_baseline))
    H, x_edges, y_edges = np.histogram2d(acc_trained, acc_baseline, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    #plt.title(" Accuracy per Expert SI-SCM vs. "+ baseline_name)
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H, cmap='YlGnBu')
    plt.axline((1,1),slope=1, ls="--", color="red")
    plt.plot(np.nanmean(acc_baseline), np.nanmean(acc_trained), 'rx', markersize=8)
    plt.xlabel('Accuracy '+baseline_name)
    plt.ylabel('Accuracy Trained SI-SCM')
    plt.colorbar()
    plt.savefig(path+"hist2d_E_"+baseline_name)
    plt.show()

def plot_diff_acc_group_2D(trained, baseline, baseline_name, n_experts):
    acc_trained = np.array([np.mean(trained[:,4]==trained[:,5], where= (trained[:,2]==exp) & (trained[:,6]==1)) for exp in range(n_experts)])
    acc_baseline = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==1)) for exp in range(n_experts)])
    print("scenario 2 - #experts not displayed: ", np.arange(n_experts,dtype=int)[np.isnan(acc_trained)])
    print("scenario 2 - #experts trained SI-SCM is more accurate for than ",baseline_name,": ", np.sum(acc_trained>acc_baseline))
    H, x_edges, y_edges = np.histogram2d(acc_trained, acc_baseline, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    #plt.title("Accuracy per Expert (group obs.) SI-SCM vs. "+ baseline_name)
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H, cmap='YlGnBu')
    plt.axline((1,1),slope=1, ls="--", color="red") 
    plt.plot(np.nanmean(acc_baseline), np.nanmean(acc_trained), 'rx', markersize=8)
    plt.xlabel('Accuracy '+baseline_name +' (Obs. within Groups)')
    plt.ylabel('Accuracy Trained SI-SCM (Obs. within Groups)')
    plt.colorbar()
    plt.savefig(path+"hist2d_GE_"+baseline_name)
    plt.show()

def plot_diff_acc_nongroup_2D(trained, baseline, baseline_name, n_experts):
    acc_trained = np.array([np.mean(trained[:,4]==trained[:,5], where= (trained[:,2]==exp) & (trained[:,6]==0)) for exp in range(n_experts)])
    acc_baseline = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==0)) for exp in range(n_experts)])
    H, x_edges, y_edges = np.histogram2d(acc_trained, acc_baseline, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    #plt.title("Accuracy per Expert (non group obs.) SI-SCM vs. "+ baseline_name)
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H, cmap='YlGnBu')
    plt.axline((1,1),slope=1, ls="--", color="red") 
    plt.plot(np.nanmean(acc_baseline), np.nanmean(acc_trained), 'rx', markersize=8)
    plt.xlabel('Accuracy '+baseline_name + ' (Obs. accross Groups)')
    plt.ylabel('Accuracy Trained SI-SCM (Obs. accross Groups)')

    plt.colorbar()
    plt.savefig(path+"hist2d_NGE_"+baseline_name)
    plt.show()

def plot_acc_gng_2D( baseline, baseline_name, n_experts):
    acc_group = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==1)) for exp in range(n_experts)])
    acc_not_group = np.array([np.mean(baseline[:,4]==baseline[:,5], where= (baseline[:,2]==exp) & (baseline[:,6]==0)) for exp in range(n_experts)])
    print("scenario 2 vs.3 - #experts ", baseline_name," more accurate in 2 than 3: ", np.sum(acc_group>acc_not_group))
    H, x_edges, y_edges = np.histogram2d(acc_group, acc_not_group, bins=(np.arange(0, 1.1, 0.025),np.arange(0, 1.1, 0.025)))
    #plt.title("Accuracy per Expert group vs. no group obs.: "+ baseline_name)
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H, cmap='YlGnBu')
    plt.axline((1,1),slope=1, ls="--", color="red")
    plt.plot(np.nanmean(acc_not_group), np.nanmean(acc_group), 'rx', markersize=8)
    plt.ylabel('Accuracy given Obs. within Groups')
    plt.xlabel('Accuracy given Obs. accross Groups')

    plt.colorbar()
    plt.savefig(path+"hist2d_EGNG_"+baseline_name)
    plt.show()

def print_overall_acc(trained, untrained, nb, gnb):
    trained_acc = np.mean(trained[:,4]==trained[:,5])
    trained_acc_g = np.mean(trained[:,4]==trained[:,5], where= trained[:,6]==1)
    trained_acc_ng = np.mean(trained[:,4]==trained[:,5], where= trained[:,6]==0)

    untrained_acc = np.mean(untrained[:,4]==untrained[:,5])
    untrained_acc_g = np.mean(untrained[:,4]==untrained[:,5], where= untrained[:,6]==1)
    untrained_acc_ng = np.mean(untrained[:,4]==untrained[:,5], where= untrained[:,6]==0)

    nb_acc = np.mean(nb[:,4]==nb[:,5])
    nb_acc_g = np.mean(nb[:,4]==nb[:,5], where= nb[:,6]==1)
    nb_acc_ng = np.mean(nb[:,4]==nb[:,5], where= nb[:,6]==0)

    gnb_acc = np.mean(gnb[:,4]==gnb[:,5])
    
    print("Acc\t:", trained_acc, untrained_acc, nb_acc, gnb_acc)
    print("Acc G\t:", trained_acc_g, untrained_acc_g, nb_acc_g)
    print("Acc NG\t:", trained_acc_ng, untrained_acc_ng, nb_acc_ng)

def main():
    latexify()
    siscm_results = pd.read_csv(path+"evaluation_results_trained.csv").to_numpy()
    naive_results = pd.read_csv(path+"evaluation_results_naive.csv").to_numpy()
    nb_baseline_results = pd.read_csv(path+"evaluation_results_nb_baseline.csv").to_numpy()
    gnb_base_results = pd.read_csv(path+"evaluation_results_base_model.csv").to_numpy()
 
    n_experts = int(np.max(siscm_results[:,2])+1)
    n_data_test = int(np.max(siscm_results[:,0])+1)
    disagreement = [np.mean((naive_results[:,4]!=naive_results[:,3]), where=naive_results[:,2]==exp) for exp in range(n_experts)]
    print("# data test: ",n_data_test)
    print("# experts: ",n_experts)
    print("Disagreement ratio: ",np.nanmean(disagreement))
    n_pred = siscm_results.shape[0]
    n_pred_group = int(np.sum(siscm_results[:,6]==1))
    n_pred_notgroup = int(np.sum(siscm_results[:,6]==0))
    print("#Pred: ", n_pred)
    print("#Pred Group: ", n_pred_group)
    print("#Pred not Group: ", n_pred_notgroup)
    pred_per_data = [np.sum((naive_results[:,0]==d)) for d in range(n_data_test)]
    print(np.mean(pred_per_data))

    print_overall_acc(siscm_results, naive_results, nb_baseline_results, gnb_base_results)
    plot_confusion_matrix(siscm_results, "Trained SI-SCM")
    plot_confusion_matrix(naive_results, "Untrained SI-SCM SI-SCM")
    plot_confusion_matrix( nb_baseline_results, "Naive Bayes Baseline")


    plot_diff_acc_2D(siscm_results, naive_results, "Untrained SI-SCM", n_experts)
    plot_diff_acc_2D(siscm_results, nb_baseline_results, "Naive Bayes Baseline", n_experts)

    plot_diff_acc_group_2D(siscm_results, naive_results, "Untrained SI-SCM", n_experts)
    plot_diff_acc_nongroup_2D(siscm_results, naive_results, "Untrained SI-SCM", n_experts)
    plot_diff_acc_group_2D(siscm_results, nb_baseline_results, "Naive Bayes Baseline", n_experts)
    plot_diff_acc_nongroup_2D(siscm_results, nb_baseline_results, "Naive Bayes Baseline", n_experts)

    plot_acc_gng_2D( nb_baseline_results, "Naive Bayes Baseline", n_experts)


if __name__ == "__main__":
    main()
