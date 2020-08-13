import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd 
import os 
import itertools
import time 
import matplotlib as mpl
from cycler import cycler
from collections import Counter
import random 
from scipy.special import comb
import multiprocessing as mp

def network_construction(network_type):
    """TODO: Docstring for network_construction.

    :network_type: TODO
    :returns: TODO

    """
    if network_type == 'Celegans':
        filename = '../celegansneural.gml'
    data = nx.read_gml(filename)
    A = nx.adjacency_matrix(data).todense()
    A_undirected = A + A.transpose()
    A_unweighted = np.heaviside(A_undirected, 0)
    M = np.sum(np.heaviside(A, 0))
    return A, A_undirected, A_unweighted

def score(A, weights, method, alpha = None, beta = None):
    """common neighbors by the weighted information 

    :snapshot_num: TODO
    :returns: TODO

    """
    N = np.size(A, -1)
    if weights == 'unweighted': 
        A_CN = np.dot(A, A)
    elif weights == 'weighted': 
        A_kj = np.transpose(np.tile(A, (N, 1, 1)), (1, 0, 2))
        A_CN = np.sum((A+A_kj) * np.heaviside(A*A_kj, 0), 2)

    elif weights == 'directed':
        A_kj = np.transpose(np.tile(A, (N, 1, 1)), (1, 0, 2))
        A_out = np.sum((A+A_kj) * np.heaviside(A*A_kj, 0), 2)
        A_indegree = np.sum(A, 1)
        in_index = np.where(A_indegree>0)
        # A_out = np.sum((A/np.tile(np.sum(A, 1), (np.size(A, 1), 1)).transpose() + A_kj/np.transpose(np.tile(np.sum(A_kj, -1), (np.size(A, 1), 1, 1)), (1, 2, 0))) * np.heaviside(A*A_kj, 0), 2)

        A_jk = np.transpose(np.tile(A, (N, 1, 1)), (2, 0, 1))
        A_in =  np.sum((A.transpose()+A_jk) * np.heaviside(A.transpose()*A_jk, 0), 2)
        A_cross =  np.sum((A.transpose()+A_kj) * np.heaviside(A.transpose()*A_kj, 0), 2) + np.sum((A+A_jk) * np.heaviside(A*A_jk, 0), 2)
        A_CN = beta * (A_out)+ (1-beta) * A_in

    "calculate closeness"
    G = nx.from_numpy_matrix(A)
    closeness_dict = nx.closeness_centrality(G)
    # closeness_dict = nx.degree_centrality(G)
    closeness = np.array(list(closeness_dict.values()))
    closeness_matrix = closeness.reshape(N, 1) * closeness


    if method == 'gravitation':
        F = closeness_matrix * A_CN * A_CN
        score = F
    elif method == 'CN':
        score = A_CN
    elif method == 'closeness':
        score = closeness_matrix
    elif method == 'CCPA':
        A_CN = A_CN/A_CN.max()
        closeness_matrix = closeness_matrix/ closeness_matrix.max()
        score = alpha * A_CN + (1-alpha) *closeness_matrix
    return score

def random_combination(iterable, r):
    """Random selection from itertools.combinations(iterable, r)
    
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def remove_edge(network_type, remove_fraction, times, weights, method, total_runs, beta):
    """TODO: Docstring for restoration.

    :arg1: TODO
    :returns: TODO

    """
    A_actual, A_undirected, A_unweighted = network_construction(network_type)
    N = np.size(A_actual, 1)
    A_unweighted_list = np.ravel(np.tril(A_unweighted))
    A_undirected_list = np.ravel(np.tril(A_undirected))
    A_actual_list = np.ravel(A_actual)

    exist_index = np.where(A_actual_list > 0)[0]
    M = np.size(exist_index) # number of edges
    m = np.arange(M)
    remove_num = round(M * remove_fraction)
    add_num = remove_num * times
    total_num = total_runs

    remove_set = np.zeros((total_num, remove_num))
    success_set = np.ones((total_num, remove_num)) * (-1)
    miss_set = np.ones((total_num, remove_num)) * (-1)
    failure_set = np.ones((total_num, add_num)) * (-1)

    for i in range(total_num):
        choice = random_combination(m, remove_num)
        A_change = A_actual_list.copy()
        remove_edges = exist_index[list(choice)]
        A_change[remove_edges] = 0
        A_matrix = A_change.reshape(N, N)
        if weights == 'unweighted':
            A_undirected = A_matrix + A_matrix.transpose()
            A = np.heaviside(A_undirected, 0)
        elif weights == 'weighted':
            A = A_matrix + A_matrix.transpose()
        if weights == 'directed':
            A = A_matrix
        "edge ij and edge ji share the same score."
        score_matrix = score(A, weights, method, beta=beta)
        "don't consider existing edges"
        score_matrix[A>0] = 0
        np.fill_diagonal(score_matrix, -1)
        sort_score = np.argsort(score_matrix, axis=None)[::-1]
        add_edges = sort_score[:int(add_num)]
        success = np.intersect1d(remove_edges, add_edges)
        miss = np.setdiff1d(remove_edges, success)
        failure = np.setdiff1d(add_edges, success)

        remove_set[i] = remove_edges
        success_set[i, :np.size(success)] = success
        miss_set[i, :np.size(miss)] = miss
        failure_set[i, :np.size(failure)] = failure

    return remove_set, success_set, miss_set, failure_set

def remove_single(A_actual_list, N, exist_index, m, remove_num, add_num, weights, method, beta):
    """TODO: Docstring for restoration.

    :arg1: TODO
    :returns: TODO

    """

    choice = random_combination(m, remove_num)
    A_change = A_actual_list.copy()
    remove_edges = exist_index[list(choice)]
    A_change[remove_edges] = 0
    A_matrix = A_change.reshape(N, N)
    if weights == 'unweighted':
        A_undirected = A_matrix + A_matrix.transpose()
        A = np.heaviside(A_undirected, 0)
    elif weights == 'weighted':
        A = A_matrix + A_matrix.transpose()
    if weights == 'directed':
        A = A_matrix
    "edge ij and edge ji share the same score."
    score_matrix = score(A, weights, method, beta=beta)
    "don't consider existing edges"
    score_matrix[A>0] = 0
    np.fill_diagonal(score_matrix, -1)
    sort_score = np.argsort(score_matrix, axis=None)[::-1]
    add_edges = sort_score[:int(add_num)]
    success = np.intersect1d(remove_edges, add_edges)
    miss = np.setdiff1d(remove_edges, success)
    failure = np.setdiff1d(add_edges, success)

    return remove_edges, success, miss, failure

def remove_parallel(network_type, remove_fraction, times, weights, method, total_runs, beta):
    """TODO: Docstring for restoration.

    :arg1: TODO
    :returns: TODO

    """
    A_actual, A_undirected, A_unweighted = network_construction(network_type)
    N = np.size(A_actual, 1)
    A_actual_list = np.ravel(A_actual)

    exist_index = np.where(A_actual_list > 0)[0]
    M = np.size(exist_index) # number of edges
    m = np.arange(M)
    remove_num = round(M * remove_fraction)
    add_num = remove_num * times
    total_num = total_runs

    remove_set = np.zeros((total_num, remove_num))
    success_set = np.ones((total_num, remove_num)) * (-1)
    miss_set = np.ones((total_num, remove_num)) * (-1)
    failure_set = np.ones((total_num, add_num)) * (-1)

    p = mp.Pool(cpu_number)
    result = p.starmap_async(remove_single, [(A_actual_list, N, exist_index, m, remove_num, add_num, weights, method, beta) for i in range(total_num)]).get()
    p.close()
    p.join()
    for i in range(total_num):
        remove_set[i], success_set[i, :np.size(result[i][1])], miss_set[i, :np.size(result[i][2])], failure_set[i, :np.size(result[i][3])] = result[i]
    return remove_set, success_set, miss_set, failure_set

def restore_success(network_type, remove_fraction, times, method, total_runs, beta_set):
    """TODO: Docstring for restore_statana.

    :remove_fraction: TODO
    :times: TODO
    :weights: TODO
    :method: TODO
    :total_runs: TODO
    :S: TODO
    :beta: TODO
    :returns: TODO

    """
    beta_size = np.size(beta_set)
    weights_set = ['directed'] * beta_size + ['weighted', 'unweighted']
    beta_combine = np.hstack((beta_set, np.array([0, 0])))
    success_rate_set = []
    success_rate_positive_set = []
    for weights, beta in zip(weights_set, beta_combine):
        t1 = time.time()
        # remove_set, success_set, miss_set, failure_set = remove_edge(network_type, remove_fraction, times, weights, method, total_runs, beta)
        remove_set, success_set, miss_set, failure_set = remove_parallel(network_type, remove_fraction, times, weights, method, total_runs, beta)
        t2 = time.time()
        print(beta, t2-t1)
        remove_num = np.size(remove_set, 1)
        remove_edges = np.unique(remove_set)
        success = np.sum(success_set != (-1), 1)
        add_num = np.size(failure_set, 1)
        success_rate_set.append(np.mean(success) / remove_num)
        success_positive = success[success > 0]
        success_rate_positive_set.append( np.mean(success_positive) /remove_num)


    plt.plot(beta_set, success_rate_set[:beta_size], 'o--', label='directed')
    plt.plot(beta_set, success_rate_set[-2] * np.ones(np.size(beta_set)), 'o--', label='weighted')
    plt.plot(beta_set, success_rate_set[-1] * np.ones(np.size(beta_set)), 'o--', label='unweighted')
    # plt.plot(beta_set, success_rate_positive_set, 'o--', label='included recall>0)')
    plt.xlabel('$\\beta$', fontsize=fs)
    plt.ylabel('average recall', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(fontsize=legendsize)
    #plt.legend(bbox_to_anchor=(0.98, 1.0))
    plt.show()
    return success_rate_set, success_rate_positive_set

ticksize = 15
legendsize = 14
fs = 20 
cpu_number = 10

network_type = 'Celegans'
method = 'CN'
weights = 'weighted'
weights = 'directed'
beta = 0
beta_set = np.arange(0, 1.1, 0.2)
remove_fraction = 0.1
times = 2
total_runs = 1000

restore_success(network_type, remove_fraction, times, method, total_runs, beta_set)
