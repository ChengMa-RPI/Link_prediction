import numpy as np
import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
from cycler import cycler
import itertools
import time 
marker = itertools.cycle(('d', 'v', 'o', '*'))

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:red', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:grey', 'tab:cyan', 'tab:green', 'tab:olive', 'tab:purple', 'black'])


data = pd.read_csv('../communication.csv', sep=";", header=None)
times = list(data.iloc[1:, 2])
node1 = np.array(list(map(int, list(data.iloc[1:, 0]))))
node2 = np.array(list(map(int, list(data.iloc[1:, 1]))))
N_total = np.size(np.unique(np.hstack((node1, node2))))
# weight = np.array(data.iloc[:, 3])
#time, node1, node2, weight = d
month = []
for t in times:
    month.append(int(t[5:7]))
    
month = np.array(month)
no_loop_index = np.where(node1 != node2)[0]
data_matrix = np.vstack((month, node1, node2))
data_no_loop = data_matrix[:, no_loop_index]
month = data_no_loop[0]

index_separate = [np.where(month == i)[0][-1] for i in range(1, 10)]

def snapshot_network(snapshot_num):
    """TODO: Docstring for snapshot_network.

    :snapshot_month: TODO
    :returns: TODO

    """

    if snapshot_num == 0:
        snap_net = data_no_loop[:, 0:index_separate[snapshot_num]]
    else:
        snap_net = data_no_loop[:, index_separate[snapshot_num-1]+1:index_separate[snapshot_num]]
    node1 = snap_net[1]
    node2 = snap_net[2]
    node_min = np.min(np.hstack((node1, node2))) 
    node_max = np.max(np.hstack((node1, node2))) 
    A = np.zeros((node_max, node_max))
    for n1, n2 in zip(node1, node2):
        A[n1-1, n2-1] += 1

    index = np.all(A==0, axis=0)&np.all(A==0, axis=1)
    A_actual = A[~index][:, ~index] 
    node_list = np.where(~index)[0] + 1
    A_undirected = A_actual + A_actual.transpose()
    A_unweighted = np.heaviside(A_undirected, 0)
    M = np.sum(A_actual>0)

    return A_actual, A_undirected, A_unweighted, node_list, M

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

def score_hd(A, weights, method, alpha = None, beta = None):
    """likelihood score based on weights and methods, only matrix calculation.

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

def score(A, weights, method, alpha = None, beta = None):
    """common neighbors by the weighted information 

    :snapshot_num: TODO
    :returns: TODO

    """
    N = np.size(A, -1)
    if weights == 'unweighted': 
        A_CN = np.dot(A, A)
    elif weights == 'weighted': 
        A_CN = np.zeros((N, N))
        for  i in range(N):
            A_CN[i] = np.sum((A[i] + A) * np.heaviside(A[i]*A, 0), 1)

    elif weights == 'directed':
        A_out = np.zeros((N, N))
        A_in = np.zeros((N, N))
        for  i in range(N):
            A_out[i] = np.sum((A[i] + A) * np.heaviside(A[i]*A, 0), 1)
            A_in[i] = np.sum((A[:, i] + A.transpose()) * np.heaviside(A[:, i]*A.transpose(), 0), 1)

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

def sort_score(A, node_list, weights, method, alpha=None, beta=None):
    """TODO: Docstring for predict_score.

    :method: TODO
    :snapshot_num: TODO
    :returns: TODO

    """
    score_matrix = score(A, weights, method, alpha, beta)
    #score_matrix = score_hd(A, weights, method, alpha, beta)
    # score_matrix[A>0] = -1
    score_matrix[(A+A.transpose())>0] = -1

    np.fill_diagonal(score_matrix, -1)
    N = np.size(score_matrix, 1)
    score_list = np.ravel(score_matrix)
    sort_index = np.argsort(score_list)[::-1]
    sort_value = np.sort(score_list)[::-1]
    row = np.floor(sort_index/N).astype(int)
    column = sort_index% N 
    node_row = node_list[row.tolist()]
    node_column = node_list[column.tolist()]
    sort_ij = np.vstack((row, column, node_row, node_column, sort_value, sort_index)).transpose()
    return sort_ij 

def predictor(snapshot_num, weights, method, n, S, alpha=None, beta=None):
    """ predict potential edges

    :snapshot_num: the network number
    :weights: weighted or unweighted
    :method: the method to predict potential edges, common neighbors, closeness centrality, or ombined methods
    :n: the fraction of edges to be added
    :alpha: the control parameter used in CCPA algorithm
    :return: the number of predicted edges and success rate
    
    """

    # A_actual, A_undirected, A_unweighted, node_list, M = snapshot_network(snapshot_num)
    A_actual, A_undirected, A_unweighted, node_list, M, _, _ = merge_weights(snapshot_num, S)

    add_num = int(round(M*n))
    if weights == 'unweighted':
        A = A_unweighted
    elif weights == 'weighted':
        A = A_undirected
    elif weights == 'directed':
        A = A_actual
    sort_matrix = sort_score(A, node_list, weights, method, alpha, beta)


    "link prediction in the next three snapshots"
    predict = 0
    exist_future = np.array([])
    for i in range(min(20, 9-snapshot_num)):
        A_i, _, _, node_i, _ = snapshot_network(snapshot_num + i)
        exist_edges = exist_edge(A_i, node_i)
        if exist_future.size:
            exist_future = np.vstack((exist_future, exist_edges))
        else:
            exist_future = exist_edges
    
    future_node_sum = np.sum(exist_future[:, 2:4], 1)
    sort_node_sum = np.sum(sort_matrix[:, 2:4],1)  
    for i, j in zip(sort_node_sum[:add_num], sort_matrix[:add_num]):
        if  i in future_node_sum:
            for possible in np.where([future_node_sum == i])[1]:
                if j[3] == exist_future[possible][2]:
                    predict += 1 
                    break
    success_rate = np.round(predict/add_num,2)
    return predict, success_rate

def reconstruct(snapshot_num):
    """construct network with all node included.

    :snapshot_num: TODO
    :returns: TODO

    """
    A_reconstruct = np.zeros((N_total, N_total))
    A, A_undirected, A_unweighted, node_list, M = snapshot_network(snapshot_num)
    N = np.size(node_list)
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
        for i in range(snapshot_num+1):
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
    M = np.sum(A_actual > 0)
    return A_actual, A_undirected, A_unweighted, node_list, M, A, np.heaviside(A, 0)

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
            t1 = time.time()
            prediction[j, i], success_rate[j, i] = predictor(snapshot_num, 'directed', method, portion, S, beta=beta)
            t2 = time.time()
            print(beta, portion, t2- t1)
        if plot == 'success':
            plt.plot(n, success_rate[:, i], '--', marker = next(marker), label=f'$\\beta=${beta}')
            plt.ylabel('Accuracy of prediction', fontsize=fs)
        elif plot =='prediction':
            plt.plot(n, prediction[:, i], '--', marker = next(marker), label=f'$\\beta=${beta}')
            plt.ylabel('Successfully predicted edges', fontsize=fs)
    prediction = np.zeros((np.size(n)))
    success_rate = np.zeros((np.size(n)))
    for portion, j in zip(n, range(np.size(n))):
        t1 = time.time()
        prediction[j], success_rate[j] = predictor(snapshot_num, 'unweighted', method, portion, S)
        t2 = time.time()
        print('unweighted', portion, t2 - t1)
    if plot == 'success':
        plt.plot(n, success_rate, '--', marker = next(marker), label='unweighted')
        plt.ylabel('Accuracy of prediction', fontsize=fs)
    elif plot =='prediction':
        plt.plot(n, prediction, '--', marker = next(marker), label='unweighted')
        plt.ylabel('Successfully predicted edges', fontsize=fs)

    for portion, j in zip(n, range(np.size(n))):
        t1 = time.time()
        prediction[j], success_rate[j] = predictor(snapshot_num, 'weighted', method, portion, S)
        t2 = time.time()
        print('weighted', portion, t2 - t1)
    if plot == 'success':
        plt.plot(n, success_rate, '--', marker = next(marker), label='weighted')
        plt.ylabel('Accuracy of prediction', fontsize=fs)
    elif plot =='prediction':
        plt.plot(n, prediction, '--', marker = next(marker), label='weighted')

        plt.ylabel('Successfully predicted edges', fontsize=fs)


    plt.xlabel('The fraction of existing edges to add', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(fontsize=legendsize)
    # plt.legend(bbox_to_anchor=(0.98, 1.0))
    #plt.show()
    return prediction, success_rate

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

def new_edge_pattern(S):
    """TODO: Docstring for compare.

    :snapshot_num: TODO
    :returns: TODO

    """
    New_edge = []
    Node = []
    A = np.zeros((9, N_total, N_total))
    count_all = np.zeros((9, 3))
    count_percentage = np.zeros((9, 3))
    with_cn = np.zeros((9, 2))
    M_num_all = np.zeros((9, 3))
    for i in range(9):
        snapshot_num = i 
        A_actual, A_undirected, A_unweighted, node_list, M, _, A[i] = merge_weights(snapshot_num, S)
    for i in range(8):
        M_unique, M_actual, M_num = motif_three(i, S)
        new_edge = np.array(np.where((A[i+1]-A[i])==1)) # +1 to make the start node as 1
        new_edge = np.array(np.where((A[-1]-A[i])==1)) # +1 to make the start node as 1
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
        print(i)
    return count_all, count_percentage, with_cn, M_num_all

def plot_motif(count_all, count_percentage, with_cn, M_num_all, plot_range, plot_type):
    """TODO: Docstring for plot_motif.

    :S: TODO
    :a: TODO
    :returns: TODO

    """
    mot_num = np.arange(3) + 1
    "plot the number of new edges with different motifs"
    for i in plot_range:
        if plot_type == 'num':
            plt.plot(mot_num, count_all[i], 'o--', label=f'{i+1}')
            plt.xlabel('motifs', fontsize=fs)
            plt.ylabel('number of new edges', fontsize=fs)
        elif plot_type == 'percent':
            plt.plot(mot_num, count_percentage[i], 'o--', label=f'{i+1}')
            plt.xlabel('motifs', fontsize=fs)
            plt.ylabel('fraction of new edges', fontsize=fs)
        elif plot_type == 'M_num':
            plt.plot(mot_num, M_num_all[i], 'o-', label=f'{i+1}')
            plt.xlabel('motifs', fontsize=fs)
            plt.ylabel('number of motifs', fontsize=fs)

    if plot_type == 'with_cn':
        plt.plot(np.arange(9) + 1, with_cn[:, 0], 'o-')
        plt.xlabel('snapshot', fontsize=fs)
        plt.ylabel('new edges with CN', fontsize=fs)


    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(fontsize=legendsize)
    #plt.legend(bbox_to_anchor=(0.98, 1.0))
    #plt.show()


ticksize = 15
legendsize = 13
fs = 20 

weights = 'weighted'
beta = 0
method = 'CN'
plot = 'success'
plot = 'prediction'
n = 0.1
n = np.arange(0.01, 0.21, 0.05)
n = np.arange(0.1, 0.51, 0.1)
beta_set = np.arange(0, 1.1, 0.2)
S = 100
S = 'inf'
snapshot_num = 6

prediction, success = beta_n(method, snapshot_num, beta_set, n, S, plot, alpha=None, beta=None)

'''
count_all, count_percentage, with_cn, M_num_all = new_edge_pattern(S)

plot_range = np.arange(7)
plot_type = 'percent'
plot_motif(count_all, count_percentage, with_cn, M_num_all, plot_range, plot_type)
plt.show()
'''
