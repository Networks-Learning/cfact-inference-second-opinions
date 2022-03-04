#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics.cluster import adjusted_rand_score

def get_fig_dim(width=900, fraction=1, aspect_ratio=1):
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
    fig_height_in = fig_width_in

    fig_dim = (fig_width_in,fig_height_in)#, golden_ratio)

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

# def overlap_coefficient(set1, set2):
#     union_size = len(set1.intersection(set2))
#     return union_size / len(set1)

def compare_groups(siscm_Psi, siscm_true):
    #looks more reasonable:
    #scatter_factor = # groups the original group was split 
    #overlap_factor = % of the original group found in each training group
    #real_group_list = siscm_true.get_group_membership_list()
    #trained_group_list = siscm_Psi.get_group_membership_list()
    # overlap_factor = []
    # overlap_per_trained = {}
    # for group in real_group_list:
    #     trained_group_index = {siscm_Psi.get_group_index(ind) for ind in group}
    #     overlap_factor.append([ overlap_coefficient(group, trained_group_list[i]) for i in trained_group_index])
    #     for i in trained_group_index:
    #         overlap_per_trained[i] = overlap_per_trained.get(i, []) + [overlap_coefficient(group, trained_group_list[i])]
    # scatter_factor = [len(l) for l in overlap_factor]
    # print()
    # print("Number of greedy groups each real group was split into")
    # print(scatter_factor)
    # print()
    # print("Percentage of the real group in each greedy group")
    # ratio_per_real = [ np.around(factor,decimals=2).tolist() for factor in overlap_factor]
    # print(ratio_per_real)
    # print()
    # print("Percentage of each real group in a greedy group")
    # ratio_per_greedy = [ np.around(factor,decimals=2).tolist() for g, factor in overlap_per_trained.items()]
    # print(ratio_per_greedy)
    # print()
    
    trained = [siscm_Psi.get_group_index(ind) for ind in range(siscm_Psi.n_experts)]
    real = [siscm_true.get_group_index(ind) for ind in range(siscm_Psi.n_experts)]
    ari_value = adjusted_rand_score(trained, real)
    print("ARI value of found partition: ",ari_value)
    return ari_value

