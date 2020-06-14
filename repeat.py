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

marker = itertools.cycle(('d', 'v', 'o', '*'))

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:red', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:grey', 'tab:cyan', 'tab:green', 'tab:olive', 'tab:purple', 'black'])

def undirect_unweight(snapshot_num):
    """convert adjacency matrix to unweighted and undirected graph

    :data: TODO
    :returns: TODO

    """

    data = np.array(pd.read_csv(f'../Datasets Caviar/CAVIAR{snapshot_num}.csv', header=None).iloc[:, :])
    node_list = data[0, 1:].astype(int)
    N = np.size(node_list)
    A = data[1:, 1:]
    A_undirected = A + np.transpose(A)
    A_unweighted = np.heaviside(A_undirected, 0)
    M = np.sum(np.heaviside(A, 0))
    M = np.sum(A_unweighted)/2

    node_index = np.argsort(node_list)
    A_unweighted = A_unweighted[node_index][:, node_index]
    A_undirected = A_undirected[node_index][:, node_index]
    return A_undirected, A_unweighted, node_list[node_index], M 

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
        A_CN = beta * (A_out )+ (1-beta) * A_in

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

def exist_edge(A, node_list):
    """TODO: Docstring for non_exist.

    :A_unweighted: TODO
    :returns: TODO

    """
    exist_rc = np.array(np.where(A > 0))
    exist_index = np.array(np.where(np.ravel(A) > 0))
    exist_ij = np.vstack((exist_rc, node_list[exist_rc[0].tolist()], node_list[exist_rc[1].tolist()], exist_index))
    # exist_undirected = exist_ij[:, exist_ij[0]> exist_ij[1]].transpose()
    return exist_ij.transpose()

def sort_score(A, node_list, weights, method, alpha=None, beta=None):
    """TODO: Docstring for predict_score.

    :method: TODO
    :snapshot_num: TODO
    :returns: TODO

    """
    score_matrix = score(A, weights, method, alpha, beta)
    N = np.size(score_matrix, 1)
    score_list = np.ravel(score_matrix)
    sort_index = np.argsort(score_list)[::-1]
    sort_value = np.sort(score_list)[::-1]
    diagonal_index = np.array([i * N + i for i in range(N)])
    sort_index_ij = np.array([element for element in sort_index if element not in diagonal_index], dtype='int')
    sort_value_ij = np.array([score for element, score in zip(sort_index, sort_value) if element not in diagonal_index]) 
    row = np.floor(sort_index_ij/N).astype(int)
    column = sort_index_ij % N 
    node_row = node_list[row.tolist()]
    node_column = node_list[column.tolist()]
    sort_ij = np.vstack((row, column, node_row, node_column, sort_value_ij, sort_index_ij))
    sort_undirected = sort_ij[:, row>column].transpose()
    return sort_undirected 

def predictor(snapshot_num, weights, method, n, S, alpha=None, beta=None):
    """ predict potential edges

    :snapshot_num: the network number
    :weights: weighted or unweighted
    :method: the method to predict potential edges, common neighbors, closeness centrality, or ombined methods
    :n: the fraction of edges to be added
    :alpha: the control parameter used in CCPA algorithm
    :return: the number of predicted edges and success rate
    
    """

    A_undirected, A_unweighted, node_list, M = undirect_unweight(snapshot_num)
    add_num = int(round(M*n))
    A_actual, A_undirected, A_unweighted, node_list, M, _, _ = merge_weights(snapshot_num, S)
    if weights == 'unweighted':
        A = A_unweighted
    elif weights == 'weighted':
        A = A_undirected
    elif weights == 'directed':
        A = A_actual
    sort_undirected = sort_score(A, node_list, weights, method, alpha, beta)
    exist_undirected = exist_edge(A, node_list)
    nonexist = np.array([element.tolist() for element in sort_undirected if element[-1] not in exist_undirected[:, -1]])

    "link prediction in the next three snapshots"
    file_store = '../data/'
    predict = 0
    exist_future = np.array([])
    for i in range(min(20, 11-snapshot_num)):
        A_i, _, node_i, _ = undirect_unweight(snapshot_num + i + 1)
        exist_edges = exist_edge(A_i, node_i)
        if exist_future.size:
            exist_future = np.vstack((exist_future, exist_edges))
        else:
            exist_future = exist_edges
    
    future_node_sum = np.sum(exist_future[:, 2:4], 1)
    nonexist_node_sum = np.sum(nonexist[:, 2:4],1)  
    for i, j in zip(nonexist_node_sum[:add_num], nonexist):
        if  i in future_node_sum:
            for possible in np.where([future_node_sum == i])[1]:
                if j[3] == exist_future[possible][2]:
                    predict += 1 
                    break
    success_rate = np.round(predict/add_num,2)
    success_df = pd.DataFrame(np.array([snapshot_num, 100*success_rate],dtype='int' ).reshape(1,2))
    if not os.path.exists(file_store + weights + method + '.csv') :
        header = True 
    else:
        header = False
    success_df.to_csv(file_store +  weights + method + '.csv', mode='a', index=True, header=header)
    return predict, success_rate

def CCPA(alpha, weights, n, S, beta=None):
    """TODO: Docstring for CCPA.

    :alpha: TODO02
    :returns: TODO

    """
    success_set = np.zeros((np.size(alpha), 10))
    for i in range(10):
        for alpha_j, j  in zip(alpha, range(np.size(alpha))):
            predict, success_set[j, i] = predictor(i+1, weights, 'CCPA', n, S, alpha=alpha_j, beta=beta)
        plt.plot(alpha, success_set[:, i], '--', marker = next(marker), label=f'{i+1}')
        plt.xlabel('$\\alpha$', fontsize=fs)
        plt.ylabel('Accuracy of prediction', fontsize=fs)
    plt.legend()

    return success_set

def forget_effect(S, weights, method, n):
    """TODO: Docstring for CCPA.

    :alpha: TODO02
    :returns: TODO

    """
    success_set = np.zeros((np.size(S), 10))
    for i in range(10):
        snapshot_num = i + 1
        for s, j  in zip(S, range(np.size(S))):
            predict, success_set[j, i] = predictor(snapshot_num, weights, method, n, s)
        plt.semilogx(S, success_set[:, i], '--', marker = next(marker), label=f'{i+1}')
        plt.xlabel('$S$', fontsize=fs)
        plt.ylabel('Accuracy of prediction', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.legend(fontsize=legendsize)
    plt.legend(bbox_to_anchor=(0.98, 1.0))
    plt.show()

    return success_set

def change_n(weights, method, n, S, plot, alpha=None, beta=None):
    """TODO: Docstring for change_n.

    :weights: TODO
    :method: TODO
    :n: TODO
    :returns: TODO

    """

    prediction = np.zeros((np.size(n), 10))
    success_rate = np.zeros((np.size(n), 10))
    for i in range(1, 8):
        for portion, j in zip(n, range(np.size(n))):
            prediction[j, i], success_rate[j, i] = predictor(i+1, weights, method, portion, S, beta=beta)
        if plot == 'success':
            plt.plot(n, success_rate[:, i], 'o-', label=f'{i+1}')
            plt.ylabel('Accuracy of prediction', fontsize=fs)
        elif plot =='prediction':
            plt.plot(n, prediction[:, i], '--', marker = next(marker), label=f'{i+1}')
            plt.ylabel('Successfully predicted edges', fontsize=fs)
    plt.xlabel('The fraction of existing edges to add', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    # plt.legend(fontsize=legendsize)
    plt.legend(bbox_to_anchor=(0.98, 1.0))
    #plt.show()
    return prediction, success_rate

def beta_n(method, snapshot_num, beta_set, n, S, plot, alpha=None, beta=None):
    """TODO: Docstring for change_n.

    :weights: TODO
    :method: TODO
    :n: TODO
    :returns: TODO

    """

    prediction = np.zeros((np.size(n), np.size(beta_set)))
    success_rate = np.zeros((np.size(n), np.size(beta_set)))
    for beta, i in zip(beta_set, range(np.size(beta_set))):
        beta = round(beta, 2)
        for portion, j in zip(n, range(np.size(n))):
            prediction[j, i], success_rate[j, i] = predictor(snapshot_num, 'directed', method, portion, S, beta=beta)
        if plot == 'success':
            plt.plot(n, success_rate[:, i], '--', marker = next(marker), label=f'$\\beta=${beta}')
            plt.ylabel('Accuracy of prediction', fontsize=fs)
        elif plot =='prediction':
            plt.plot(n, prediction[:, i], '--', marker = next(marker), label=f'$\\beta=${beta}')
            plt.ylabel('Successfully predicted edges', fontsize=fs)
    plt.xlabel('The fraction of existing edges to add', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(fontsize=legendsize)
    # plt.legend(bbox_to_anchor=(0.98, 1.0))
    #plt.show()
    return prediction, success_rate

def remove_edge(snapshot_num, remove_fraction, times, weights, method, total_runs):
    """TODO: Docstring for restoration.

    :arg1: TODO
    :returns: TODO

    """
    A_undirected, A_unweighted, node_list, M = undirect_unweight(snapshot_num)  
    N = np.size(A_undirected, 1)
    A_unweighted_list = np.ravel(np.tril(A_unweighted))
    A_undirected_list = np.ravel(np.tril(A_undirected))

    exist_index = np.where(A_unweighted_list == 1)[0]
    M = np.size(exist_index) # number of edges
    remove_num = round(M * remove_fraction)
    add_num = remove_num * times
    m = np.arange(M)
    comb = list(itertools.combinations(m, remove_num))
    np.random.shuffle(comb)
    
    total_possible = len(comb)
    total_num = min(total_possible, total_runs)

    remove_set = np.zeros((total_num, remove_num))
    success_set = np.ones((total_num, remove_num)) * (-1)
    miss_set = np.ones((total_num, remove_num)) * (-1)
    failure_set = np.ones((total_num, add_num)) * (-1)

    for choice, i in zip(comb[: total_num], range(total_num)):
        A_change = A_undirected_list.copy()
        remove_edges = exist_index[list(choice)]
        A_change[remove_edges] = 0
        A_matrix = A_change.reshape(N, N)
        if weights == 'unweighted':
            A = np.heaviside(A_matrix, 0)
        elif weights == 'weighted':
            A = A_matrix
        score_matrix = score(A, weights, method)
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

def restoration_all(remove_fraction, times, weights, method, total_runs):
    """TODO: Docstring for restoration_all.
    :returns: TODO

    """

    remove_set = []
    success_set = []
    miss_set =[]
    failure_set = []
    Time = []
    for i in range(11):
        t1 = time.time()
        remove, success, miss, failure = remove_edge(i+1, remove_fraction, times, weights, method, total_runs)
        t2 = time.time()
        print(i, t2-t1)
        remove_set.append(remove)
        success_set.append(success)
        miss_set.append(miss)
        failure_set.append(failure)
        Time.append(t2-t1)
    return remove_set, success_set, miss_set, failure_set,  Time

def reconstruct(snapshot_num):
    """construct network with all node included.

    :snapshot_num: TODO
    :returns: TODO

    """
    N_total = 110
    A_reconstruct = np.zeros((N_total, N_total))
    data = np.array(pd.read_csv(f'../Datasets Caviar/CAVIAR{snapshot_num}.csv', header=None).iloc[:, :])
    node_list = data[0, 1:].astype(int)
    N = np.size(node_list)
    A = data[1:, 1:]
    index_sort = np.argsort(node_list)
    A_order = A[index_sort][:, index_sort]
    node_order = node_list[index_sort]
    A_reconstruct[np.ix_(node_order-1, node_order-1)] = A_order
    A_reconstruct[np.ix_(node_list-1, node_list-1)] = A
    return A_reconstruct

def merge_weights(snapshot_num, S):
    """TODO: Docstring for Merge_weights.

    :snapshot_num: TODO
    :S: TODO
    :returns: TODO

    """
    A = 0
    if S == 0:
        A = reconstruct(snapshot_num)
    else:
        for i in range(1, snapshot_num + 1, 1):
            if S == 'inf':
                A += reconstruct(i) 
            else:
                A += reconstruct(i) * np.exp( -(snapshot_num - i)/S)
                # A += reconstruct(i) * S ** (abs(snapshot_num - i))
    index = np.all(A==0, axis=0)&np.all(A==0, axis=1)
    A_actual = A[~index][:, ~index] 
    node_list = np.where(~index)[0] + 1
    A_undirected = A_actual + np.transpose(A_actual)
    A_unweighted = np.heaviside(A_undirected, 0)
    M = np.sum(A_unweighted)/2
    return A_actual, A_undirected, A_unweighted, node_list, M, A, np.heaviside(A, 0)

def compare_two_snanpshots(snapshot_num):
    """find out four categories of edges in the next snapshot: new, disappear, old, never appear.

    :snapshot_num: TODO
    :returns: TODO

    """
    A_before = np.heaviside(reconstruct(snapshot_num), 0)
    A_after = np.heaviside(reconstruct(snapshot_num+1), 0)
    diff = A_after - A_before
    exist = np.array(np.where(A_before == 1))
    # nonexist = np.array(np.where(A_before == 0))
    new = np.array(np.where(diff == 1))
    disappear = np.array(np.where(diff == -1))
    no_change = np.array(np.where(diff == 0))
    old = np.array([x for x in set.intersection(set(tuple(x) for x in exist.transpose()) ,set(tuple(x) for x in no_change.transpose()))]).transpose()
    # never_appear = np.array([x for x in set.intersection(set(tuple(x) for x in nonexist.transpose()) ,set(tuple(x) for x in no_change.transpose()))]) 
    return new, disappear, old

def new_edges(S=100):
    """TODO: Docstring for compare.

    :snapshot_num: TODO
    :returns: TODO

    """
    New_edge = []
    Node = []
    A = np.zeros((11, 110, 110))
    for i in range(11):
        snapshot_num = i + 1
        A_actual, A_undirected, A_unweighted, node_list, M, _, A[i] = merge_weights(snapshot_num, S)
        if i > 0:
            new_edge = np.array(np.where((A[i]-A[i-1])==1))

            New_edge.append(new_edge)
            node_in_edges = np.array(list(Counter(np.hstack((new_edge[0], new_edge[1]))).items()))
            Node.append(node_in_edges)
    total_new_edge = np.array(np.where((A[10] - A[0]) ==1))
    node_total = np.array(list(Counter(np.hstack((total_new_edge[0], total_new_edge[1]))).most_common()))

    for i in range(10):
        plt.plot(New_edge[i][0], New_edge[i][1], 'o', label=f'{i+1}')
    plt.xlabel('node sequence', fontsize=fs)
    plt.ylabel('node sequence', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.legend(fontsize=legendsize)
    plt.legend(bbox_to_anchor=(0.98, 1.0))
    plt.show()

    for i in range(10):
        plt.plot(Node[i][:, 0], Node[i][:, 1], 'o', label=f'{i+1}')
    plt.xlabel('node sequence', fontsize=fs)
    plt.ylabel('new edges', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.legend(fontsize=legendsize)
    plt.legend(bbox_to_anchor=(0.98, 1.0))
    #plt.show()

    "calculate closeness centrality"
    G = nx.from_numpy_matrix(A[9])
    closeness_dict = nx.closeness_centrality(G)
    closeness_dict_order = {k: v for k, v in sorted(closeness_dict.items(), key=lambda item: item[1], reverse=True)}
    closeness = np.array(list(closeness_dict_order.items()))

    "calculate degree centrality"
    degree_dict = nx.degree_centrality(G)
    degree_dict_order = {k: v for k, v in sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)}
    degree = np.array(list(degree_dict_order.items()))

    "calculate betweenness centrality"
    between_dict = nx.betweenness_centrality(G)
    between_dict_order = {k: v for k, v in sorted(between_dict.items(), key=lambda item: item[1], reverse=True)}
    between = np.array(list(between_dict_order.items()))


    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('node sequence', fontsize=fs)
    ax1.set_ylabel('total new edges', fontsize = fs)
    ax1.plot(node_total[:5, 0], node_total[:5, 1], 'o', color=color)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=ticksize)
    ax1.tick_params(axis='x', labelsize=ticksize)

    color = 'tab:blue'
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('closeness centrality', fontsize = fs)  # we already handled the x-label with ax1
    ax2.plot(closeness[:5, 0], closeness[:5, 1], '*', color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize = ticksize)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return node_total, closeness, degree, between
        
def motif(snapshot_num, S, a):
    """statistic analysis for different motifs

    :snapshot_num: TODO
    :S: memory decay parameter
    :a: edge polarization, no unidirection a = 0, no bidirection a = 1 
    :returns: TODO

    """
    M1 = set()
    M2 = set()
    M3 = set()
    M4 = set()
    M5 = set()
    M6 = set()
    M7 = set()
    M8 = set()
    M9 = set()
    M10 = set()
    M11 = set()
    M12 = set()
    M13 = set()
    A, A_undirected, A_unwt, node_list, M, A_full, A_full_unweighted = merge_weights(snapshot_num, S)
    A_polar = np.zeros((np.size(A, 1), np.size(A, 1)))
    A_balance = np.zeros((np.size(A, 1), np.size(A, 1)))
    A_add = A + A.transpose()
    A_subtract = A - A.transpose()
    index_nonzero = np.where(A_add > 0)
    A_polar[index_nonzero] = A_subtract[index_nonzero] / A_add[index_nonzero]  # balance degree
    A_balance[index_nonzero] = 1 - abs(A_subtract[index_nonzero] / A_add[index_nonzero])  # balance degree
    A_bi = np.heaviside(A_balance -a, 0)
    A_pos = np.heaviside(A_polar - (1-a), 1)
    bi_num = np.sum(A_bi>0, 1)
    for i in np.where(bi_num>1)[0]:
        index_jk = np.where(A_bi[i]>0)[0]
        jk = list(itertools.combinations(index_jk, 2))
        for j, k in jk:
            if A_bi[j, k] > 0:
                M1.add((i, j, k))
            elif A_pos[j, k] > 0:
                M2.add((i, k, j))
            elif A_pos[k, j] > 0:
                M2.add((i, j, k))
            elif A[j, k] == 0 and A[k, j ] ==0:
                M8.add((i, j, k))

    for i in np.where(bi_num >= 1)[0]:
        for j in np.where(A_bi[i]>0)[0]:
            if np.sum(A_pos[i] > 0) > 0:
                for k in np.where(A_pos[i] > 0)[0]:
                    if A_bi[j, k] == 0 and A_pos[k, j] > 0:
                        M3.add((i, j, k))
                    elif A_bi[j, k] == 0 and A_pos[j, k] > 0:
                        M4.add((i, j, k))
                    elif A[j, k] == 0 and A[k, j] == 0:
                        M9.add((i, j, k))

            if np.sum(A_pos[:, i] > 0) > 0:
                for k in np.where(A_pos[:, i] > 0)[0]:
                    if A_bi[j, k] == 0 and A_pos[k, j] > 0:
                        M5.add((i, j, k))
                    elif A_bi[j, k] == 0 and A_pos[j, k] > 0:
                        M3.add((j, i, k))
                    elif A[j, k] == 0 and A[k, j] == 0:
                        M10.add((i, j, k))

    for i in range(np.size(A, 1)):
        if np.sum(A_pos[i] > 0) > 1:
            index_jk = np.where(A_pos[i]>0)[0]
            jk = list(itertools.combinations(index_jk, 2))
            for j, k in jk:
                if A_bi[j, k] == 0 and A_pos[j, k] > 0:
                    M7.add((i, j, k))
                elif A_bi[j, k] == 0 and A_pos[k, j] > 0:
                    M7.add((i, k, j))
                elif A[j, k] == 0 and A[k, j] ==0:
                    M11.add((i, j, k))
        if np.sum(A_pos[:, i] > 0) > 1:
            index_jk = np.where(A_pos[:, i]>0)[0]
            jk = list(itertools.combinations(index_jk, 2))
            for j, k in jk:
                if A[j, k] == 0 and A[k, j] ==0:
                    M12.add((i, j, k))
        if np.sum(A_pos[i] > 0) >= 1:
            for j in np.where(A_pos[i]>0)[0]:
                if np.sum(A_pos[j] > 0) > 0:
                    for k in np.where(A_pos[j] > 0)[0]:
                        if A_bi[i, k] == 0 and A_pos[k, i] > 0:
                            M6.add((i, j, k))
                        elif A[k, i] == 0 and A[i, k] ==0:
                            M13.add((i, j, k))
    "delete duplicate elements"
    M = (M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13)
    M_unique = []
    M_actual = []
    M_num = []
    for i in M:
        if len(i):
            unique = set(tuple(sorted(mot)) for mot in i)
            M_unique.append(unique)
            M_actual.append(node_list[np.array(list(unique))])
            M_num.append(len(unique))
        else:
            M_actual.append([])
            M_unique.append(set())
            M_num.append(0)

    return M_unique, M_actual, np.array(M_num)

def motif_three(snapshot_num, S):
    """TODO: Docstring for motif_three.

    :arg1: TODO
    :returns: TODO

    """
    M1 = set()
    M2 = set()
    M3 = set()
    A, A_undirected, A_unwt, node_list, M, A_full, A_full_unweighted = merge_weights(snapshot_num, S)
    A_in = np.sum(A>0, 1)  # caller
    A_out = np.sum(A>0, 0)  # receiver
    for i in np.where(A_in>=2)[0]:
        index_jk = np.where(A[i]>0)[0]
        jk = list(itertools.combinations(index_jk, 2))
        for j, k in jk:
            M1.add((i, j, k))
    for i in np.where(A_out>=2)[0]:
        index_jk = np.where(A[:, i]>0)[0]
        jk = list(itertools.combinations(index_jk, 2))
        for j, k in jk:
            M2.add((i, j, k))

    for i in np.where(A_in>=1)[0]:
        for j in np.where(A[i]>0)[0]:
            for k in np.where(A[j]>0)[0]:
                M3.add((i, j, k))

    M = (M1, M2, M3)
    M_unique = []
    M_actual = []
    M_num = []
    for i in M:
        if len(i):
            unique = set(tuple(sorted(mot)) for mot in i)
            M_unique.append(unique)
            M_actual.append(node_list[np.array(list(unique))])
            M_num.append(len(unique))
        else:
            M_actual.append([])
            M_unique.append(set())
            M_num.append(0)

    return M_unique, M_actual, np.array(M_num)
        
def new_edge_pattern(S, a):
    """TODO: Docstring for compare.

    :snapshot_num: TODO
    :returns: TODO

    """
    New_edge = []
    Node = []
    A = np.zeros((11, 110, 110))
    count_all = np.zeros((10, 13))
    count_percentage = np.zeros((10, 13))
    with_cn = np.zeros((10, 2))
    M_num_all = np.zeros((10, 13))
    for i in range(11):
        snapshot_num = i + 1
        A_actual, A_undirected, A_unweighted, node_list, M, _, A[i] = merge_weights(snapshot_num, S)
    for i in range(10):
        M_unique, M_actual, M_num = motif(i+1, S, a)
        # M_unique, M_actual, M_num = motif_three(i+1, S)
        new_edge = np.array(np.where((A[i+1]-A[i])==1)) # +1 to make the start node as 1
        new_edge = np.array(np.where((A[10]-A[i])==1)) # +1 to make the start node as 1
        AAT = A[i] + A[i].transpose() 
        count = np.zeros((np.size(new_edge, 1), np.size(M_actual)))
        for n_i, n_j, n in zip(new_edge[0], new_edge[1], range(np.size(new_edge[0]))):
            node_k = np.where(AAT[n_i] * AAT[n_j] != 0)[0]  #find the common neighbors of node i and j.

            for n_k in node_k:
                triangle = sorted([n_i+1, n_j+1, n_k+1])
                for m_actual, m in zip(M_actual, range(len(M_actual))):
                    if np.size(m_actual) and max(np.sum(triangle == m_actual, -1)) == 3:
                        count[n, m] += 1
                        break
        count_all[i] = np.sum(count, 0)
        index = np.where(M_num > 0)
        count_percentage[i, index[0]] = np.sum(count, 0)[index]/ M_num[index]
        with_cn[i, 0] = np.size(np.where(np.sum(count, 1) > 0)[0])/ np.size(count, 0)
        with_cn[i, 1] = np.size(np.where(np.sum(count, 1) > 0)[0])
        M_num_all[i] = M_num
    return count_all, count_percentage, with_cn, M_num_all

def plot_motif(S, a, plot_range, plot_type):
    """TODO: Docstring for plot_motif.

    :S: TODO
    :a: TODO
    :returns: TODO

    """
    count_all, count_percentage, with_cn, M_num_all = new_edge_pattern(S, a)
    mot_num = np.arange(13) + 1
    "plot the number of new edges with different motifs"
    for i in plot_range:
        if plot_type == 'num':
            plt.plot(mot_num, count_all[i], 'o--', label=f'{i+1}')
            plt.xlabel('motifs', fontsize=fs)
            plt.ylabel('number of new edges', fontsize=fs)
        elif plot_type == 'percent':
            plt.plot(mot_num, count_percentage[i], 'o--', label=f'{i+1}')
            plt.xlabel('motifs', fontsize=fs)
            plt.ylabel('percentage of new edges', fontsize=fs)
        elif plot_type == 'M_num':
            plt.plot(mot_num, M_num_all[i], 'o-', label=f'{i+1}')
            plt.xlabel('motifs', fontsize=fs)
            plt.ylabel('number of motifs', fontsize=fs)

    if plot_type == 'with_cn':
        plt.plot(np.arange(10) + 1, with_cn[:, 0], 'o-')
        plt.xlabel('snapshot', fontsize=fs)
        plt.ylabel('new edges with CN', fontsize=fs)


    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(fontsize=legendsize)
    #plt.legend(bbox_to_anchor=(0.98, 1.0))
    #plt.show()

def illu_motif():
    """TODO: Docstring for plot_motif.
    :returns: TODO

    """
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    pos = [[0, 0], [2, 0], [1, np.sqrt(3)]]
    node = [0, 1, 2]
    edge1 = [[0, 1], [1, 0], [0, 2], [2, 0], [1,2], [2, 1]]
    edge2 = [[0, 1], [1, 0], [0, 2], [2, 0], [1,2]]
    edge3 = [[0, 1], [1, 0], [2, 0], [1, 2]]
    edge4 = [[0, 1], [1, 0], [2, 0], [2, 1]]
    edge5 = [[0, 1], [1, 0], [0, 2], [1, 2]]
    edge6 = [[1, 0], [0, 2], [2, 1]]
    edge7 = [[1, 0], [2, 0], [2, 1]]
    edge8 = [[0, 1], [1, 0], [2, 0], [0, 2]]
    edge9 = [[0, 1], [1, 0], [2, 0]]
    edge10 = [[0, 1], [1, 0], [0, 2]]
    edge11 = [[1, 0], [2, 0]]
    edge12 = [[0, 1], [0, 2]]
    edge13 = [[1, 0], [2, 1]]
    edge_set = (edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8, edge9, edge10, edge11, edge12, edge13)
    for edge in edge_set[9:]:
        G = nx.DiGraph()
        G.add_nodes_from(node)
        G.add_edges_from(edge)
        nx.draw_networkx(G, pos, arrowsize=60, node_size=3000, node_color='b', width=3, labels={0:'$i$', 1:'$j$', 2:'$k$'}, font_size=30, font_color='r')
        limits = plt.axis('off') 
        plt.subplots_adjust(left=0.02, right=0.98, wspace=0.25, hspace=0.25, bottom=0.02, top=0.98)
        plt.show()

ticksize = 15
legendsize = 14
fs = 20 

a = 0.99999
S = 'inf'
S = 100
plot_range = np.arange(2, 9, 1)
plot_type = 'M_num'
plot_type = 'num'
plot_type = 'percent'
# motif(2, 0, a)
# visualization(2, 0)
# new_edge_pattern(S, a)
# plot_motif(S, a, plot_range, plot_type)
illu_motif()


method = 'CCPA'
method = 'CN'
weights = 'unweighted'
weights = 'directed'
alpha = np.arange(0,1.1,0.1)
beta = 0
beta_set = np.arange(0, 1, 0.2)
n = np.arange(0.1, 0.6, 0.1)
plot = 'prediction'
plot = 'success'
remove_fraction = 0.1
times = 2
total_runs = 10000
# success_set = CCPA(alpha, weights, 0.25, S)
# change_n(weights, method, n, S, plot, beta=beta)
# beta_n(method, 4, beta_set, n, S, plot, beta=beta)

# CCPA(alpha, weights, 0.25, S, beta=beta)
# forget_effect(np.arange(0,100,10), weights, method, 0.25)

#remove_set, success_set, miss_set, failure_set, Time = restoration_all(remove_fraction, times, weights, method, total_runs)


success_rate_set = []
success_rate_positive_set = []
individual_rate_set = []
success_num = []
total_num = []
for i in range(11):
    remove = remove_set[i]
    remove_num = np.size(remove, 1)
    remove_edges = np.unique(remove)
    individual = np.zeros((np.size(remove_edges)))
    for remove_one, j in zip(remove_edges, range(np.size(remove_edges))):
        index = np.where(remove_one == remove)[0]
        individual_set = success_set[i][index]
        individual_success = np.sum(individual_set != (-1), 1)
        individual[j] = np.mean(individual_success) /remove_num 
    individual_rate_set.append(individual)
    success = np.sum(success_set[i] != (-1), 1)
    add_num = np.size(failure_set[i], 1)
    success_rate_set.append(np.mean(success) / remove_num)
    success_positive = success[success > 0]
    success_rate_positive_set.append( np.mean(success_positive) /remove_num)
    success_label = success_set[i][success_set[i] != (-1)]
    miss_label = miss_set[i][miss_set[i] != (-1)]
    failure_label = failure_set[i][failure_set[i] != (-1)]
    success_occurence = np.array(list(Counter(success_label).items()))
    success_num.append(np.size(success_occurence, 0))
    total_num.append(np.size(remove_edges))



plt.plot(np.arange(11)+1, success_num, 'o--', label='restored edges')
plt.plot(np.arange(11)+1, total_num, 'o--', label='total edges')
plt.xlabel('Snapshot', fontsize=fs)
plt.ylabel('Number of edges', fontsize=fs)
plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.legend(fontsize=legendsize)
#plt.legend(bbox_to_anchor=(0.98, 1.0))
plt.show()

plot_min = np.min([np.min(individual_rate_set[i]) for i in range(11)])
plot_max = np.max([np.max(individual_rate_set[i]) for i in range(11)])

for i in range(1):
    hist, bins = np.histogram(individual_rate_set[i], 50, (plot_min, plot_max))
    plt.plot(bins[:-1], hist, '.-', label=f'{i+1}')

plt.xlabel('$P$', fontsize=fs)
plt.ylabel('Number of edges', fontsize=fs)
plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.legend(fontsize=legendsize)
#plt.legend(bbox_to_anchor=(0.98, 1.0))
plt.show()





