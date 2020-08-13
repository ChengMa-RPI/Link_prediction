import numpy as np
import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
from cycler import cycler
import itertools
import time 
import collections
import sklearn.metrics as skmetrics
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import scikitplot as skplt
marker = itertools.cycle(('d', 'v', 'o', '*'))
linestyle = itertools.cycle(('-', '-', '-.', '-.', '--', '--', ':', ':')) 
linewidth = itertools.cycle((3.5, 1.5, 3.5, 1.5, 3.5, 1.5, 3.5, 1.5)) 


mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:red', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:grey', 'tab:cyan', 'tab:green', 'tab:olive', 'tab:purple', 'black'])

def load_call():
    data = pd.read_csv('../call-Reality.txt', sep=" ", header=None)
    times = np.array(list(map(int, list(data.iloc[:, 2]))))
    node1 = np.array(list(map(int, list(data.iloc[:, 0]))))
    node2 = np.array(list(map(int, list(data.iloc[:, 1]))))
    N_order = np.sort(np.unique(np.hstack((node1, node2))))
    N_total = np.size(N_order)
    N_list = np.arange(N_total) + 1
    node1_order = []
    node2_order = []
    for i in node1:
        index = np.where(i == N_order)[0][0]
        node1_order.append(index + 1)
    for i in node2:
        index = np.where(i == N_order)[0][0]
        node2_order.append(index + 1)

    data_matrix = np.vstack((times, node1_order, node2_order))

    separate_num = 10
    time_seperate = np.linspace(np.min(times), np.max(times), separate_num+1)
    data_order = np.vstack((np.array([1]), data_matrix[1:, np.where(times == np.min(times))[0]].reshape(2,1)))
    month = 0
    for i in range(separate_num):
        month = month + 1 
        index = np.where((times> time_seperate[i]) &(times<=time_seperate[i+1]))[0]
        if np.size(index) == 0:
            month = month - 1
        data_order = np.hstack((data_order, np.vstack((np.ones(np.size(index), dtype=int) * month, data_matrix[1:, index]))))


    no_loop_index = np.where(data_order[1] != data_order[2])[0]
    data_no_loop = data_order[:, no_loop_index]
    month = data_no_loop[0]

    index_separate = [np.where(month == i)[0][-1] for i in range(np.min(np.unique(month)), np.max(np.unique(month)) +1 )]
    return data_no_loop, index_separate, N_total

def load_core_call():
    data = pd.read_csv('../call-Reality.txt', sep=" ", header=None)
    node_core = np.array(pd.read_csv('../core-call-Reality.txt', sep=" ", header=None).iloc[:, 0])
    times = np.array(list(map(int, list(data.iloc[:, 2]))))
    node1 = np.array(list(map(int, list(data.iloc[:, 0]))))
    node2 = np.array(list(map(int, list(data.iloc[:, 1]))))

    node1_core = np.nonzero(np.in1d(node1, node_core))[0]
    node2_core = np.nonzero(np.in1d(node2, node_core))[0]
    core_index = np.array(list(set(node1_core).intersection(node2_core)))
    data_matrix = np.vstack((times, node1, node2))[:,core_index]

    N_order = np.sort(np.unique(np.hstack((data_matrix[1], data_matrix[2]))))
    N_total = np.size(N_order)
    node1_order = []
    node2_order = []
    for i in data_matrix[1]:
        index = np.where(i == N_order)[0][0]
        node1_order.append(index + 1)
    for i in data_matrix[2]:
        index = np.where(i == N_order)[0][0]
        node2_order.append(index + 1)

    data_matrix = np.vstack((data_matrix[0], node1_order, node2_order))
    times = data_matrix[0]

    separate_num = 10
    time_seperate = np.linspace(np.min(times), np.max(times), separate_num+1)
    data_order = np.vstack((np.array([1]), data_matrix[1:, np.where(times == np.min(times))[0]].reshape(2,1)))
    month = 0
    for i in range(separate_num):
        month = month + 1 
        index = np.where((times> time_seperate[i]) &(times<=time_seperate[i+1]))[0]
        if np.size(index) == 0:
            month = month - 1
        data_order = np.hstack((data_order, np.vstack((np.ones(np.size(index), dtype=int) * month, data_matrix[1:, index]))))


    no_loop_index = np.where(data_order[1] != data_order[2])[0]
    data_no_loop = data_order[:, no_loop_index]
    month = data_no_loop[0]

    index_separate = [np.where(month == i)[0][-1] for i in range(np.min(np.unique(month)), np.max(np.unique(month)) +1 )]
    return data_no_loop, index_separate, N_total

def load_most_call():
    data = pd.read_csv('../call-Reality.txt', sep=" ", header=None)
    node_core = np.array(pd.read_csv('../core-call-Reality.txt', sep=" ", header=None).iloc[:, 0])
    times = np.array(list(map(int, list(data.iloc[:, 2]))))
    node1 = np.array(list(map(int, list(data.iloc[:, 0]))))
    node2 = np.array(list(map(int, list(data.iloc[:, 1]))))
    node_core = np.array(collections.Counter(np.hstack((node1, node2))).most_common())[:100, 0]

    node1_core = np.nonzero(np.in1d(node1, node_core))[0]
    node2_core = np.nonzero(np.in1d(node2, node_core))[0]
    core_index = np.array(list(set(node1_core).intersection(node2_core)))
    data_matrix = np.vstack((times, node1, node2))[:,core_index]

    N_order = np.sort(np.unique(np.hstack((data_matrix[1], data_matrix[2]))))
    N_total = np.size(N_order)
    node1_order = []
    node2_order = []
    for i in data_matrix[1]:
        index = np.where(i == N_order)[0][0]
        node1_order.append(index + 1)
    for i in data_matrix[2]:
        index = np.where(i == N_order)[0][0]
        node2_order.append(index + 1)

    data_matrix = np.vstack((data_matrix[0], node1_order, node2_order))
    times = data_matrix[0]

    separate_num = 10
    time_seperate = np.linspace(np.min(times), np.max(times), separate_num+1)
    data_order = np.vstack((np.array([1]), data_matrix[1:, np.where(times == np.min(times))[0]].reshape(2,1)))
    month = 0
    for i in range(separate_num):
        month = month + 1 
        index = np.where((times> time_seperate[i]) &(times<=time_seperate[i+1]))[0]
        if np.size(index) == 0:
            month = month - 1
        data_order = np.hstack((data_order, np.vstack((np.ones(np.size(index), dtype=int) * month, data_matrix[1:, index]))))


    no_loop_index = np.where(data_order[1] != data_order[2])[0]
    data_no_loop = data_order[:, no_loop_index]
    month = data_no_loop[0]

    index_separate = [np.where(month == i)[0][-1] for i in range(np.min(np.unique(month)), np.max(np.unique(month)) +1 )]
    return data_no_loop, index_separate, N_total

def load_ia_call():
    data = pd.read_csv('../ia-reality-call.edges', sep=",", header=None)
    times = np.array(list(map(int, list(data.iloc[:, 2]))))
    node1 = np.array(list(map(int, list(data.iloc[:, 0]))))
    node2 = np.array(list(map(int, list(data.iloc[:, 1]))))
    N_order = np.sort(np.unique(np.hstack((node1, node2))))
    N_total = np.size(N_order)
    N_list = np.arange(N_total) + 1
    node1_order = []
    node2_order = []
    for i in node1:
        index = np.where(i == N_order)[0][0]
        node1_order.append(index + 1)
    for i in node2:
        index = np.where(i == N_order)[0][0]
        node2_order.append(index + 1)

    data_matrix = np.vstack((times, node1_order, node2_order))

    separate_num = 10
    time_seperate = np.linspace(np.min(times), np.max(times), separate_num+1)
    data_order = np.vstack((np.array([1]), data_matrix[1:, np.where(times == np.min(times))[0]].reshape(2,1)))
    month = 0
    for i in range(separate_num):
        month = month + 1 
        index = np.where((times> time_seperate[i]) &(times<=time_seperate[i+1]))[0]
        if np.size(index) == 0:
            month = month - 1
        data_order = np.hstack((data_order, np.vstack((np.ones(np.size(index), dtype=int) * month, data_matrix[1:, index]))))


    no_loop_index = np.where(data_order[1] != data_order[2])[0]
    data_no_loop = data_order[:, no_loop_index]
    month = data_no_loop[0]

    index_separate = [np.where(month == i)[0][-1] for i in range(np.min(np.unique(month)), np.max(np.unique(month)) +1 )]
    return data_no_loop, index_separate, N_total

def load_email():
    data = pd.read_csv('../communication.csv', sep=";", header=None)
    times = list(data.iloc[1:, 2])
    node1 = np.array(list(map(int, list(data.iloc[1:, 0]))))
    node2 = np.array(list(map(int, list(data.iloc[1:, 1]))))
    N_total = np.size(np.unique(np.hstack((node1, node2))))
    month = []
    for t in times:
        month.append(int(t[5:7]))
        
    month = np.array(month)
    no_loop_index = np.where(node1 != node2)[0]
    data_matrix = np.vstack((month, node1, node2))
    data_no_loop = data_matrix[:, no_loop_index]
    month = data_no_loop[0]

    index_separate = [np.where(month == i)[0][-1] for i in range(np.min(np.unique(month)), np.max(np.unique(month)) +1 )]
    return data_no_loop, index_separate, N_total

def load_message():
    data = pd.read_csv('../OCnodeslinks.txt', sep=" ", header=None)
    times = list(data.iloc[:, 0])
    node1 = np.array(data.iloc[:, 1])
    node2 = np.array(data.iloc[:, 2])
    N_total = np.size(np.unique(np.hstack((node1, node2))))
    weight = np.array(data.iloc[:, 3])
    month = []
    for t in times:
        month.append(int(t[5:7]))
        
    month = np.array(month)
    no_loop_index = np.where(node1 != node2)[0]
    data_matrix = np.vstack((month, node1, node2, weight))
    data_no_loop = data_matrix[:, no_loop_index]
    month = data_no_loop[0]

    index_separate = [np.where(month == i)[0][-1] for i in range(np.min(np.unique(month)), np.max(np.unique(month)) +1 )]
    return data_no_loop, index_separate, N_total

def snapshot_network(snapshot_num):
    """TODO: Docstring for snapshot_network.

    :snapshot_month: TODO
    :returns: TODO

    """

    if network_type == 'Caviar':
        data = np.array(pd.read_csv(f'../Datasets Caviar/CAVIAR{snapshot_num}.csv', header=None).iloc[:, :])
        node_list = data[0, 1:].astype(int)
        N = np.size(node_list)
        A = data[1:, 1:]
        A_undirected = A + np.transpose(A)
        A_unweighted = np.heaviside(A_undirected, 0)
        M = np.sum(np.heaviside(A, 0))
        node_index = np.argsort(node_list)
        A_actual = A[node_index][:, node_index]
        A_unweighted = A_unweighted[node_index][:, node_index]
        A_undirected = A_undirected[node_index][:, node_index]
        node_list = node_list[node_index]

    else:
        if snapshot_num == 0:
            snap_net = data_no_loop[:, 0:index_separate[snapshot_num]]
        else:
            snap_net = data_no_loop[:, index_separate[snapshot_num-1]+1:index_separate[snapshot_num]]
        node1 = snap_net[1]
        node2 = snap_net[2]
        if np.size(snap_net, 0) == 4:
            weight = snap_net[3]
        else:
            weight = np.ones((np.size(snap_net, 1)))
        node_min = np.min(np.hstack((node1, node2))) 
        node_max = np.max(np.hstack((node1, node2))) 
        A = np.zeros((node_max, node_max))
        for n1, n2, w in zip(node1, node2, weight):
            A[n1-1, n2-1] += w

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

    # A_actual, A_undirected, A_unweighted, node_list, M, month_num = snapshot_network(snapshot_num)
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
    
    A_actual, A_undirected, A_unweighted, node_list, M, _, _ = merge_weights(month_num-1, S)
    exist_future = exist_edge(A_actual, node_list)
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
    A, A_undirected, A_unweighted, node_list, M = snapshot_network(snapshot_num)
    A_reconstruct = np.zeros((N_total, N_total))
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
                A_temp = reconstruct(i) 
                A += A_temp
            else:
                A_temp = reconstruct(i) 
                A += A_temp * np.exp( -(snapshot_num - i)/S) 
                # A += reconstruct(i) * S ** (abs(snapshot_num - i))
    index = np.all(A==0, axis=0)&np.all(A==0, axis=1)
    A_actual = A[~index][:, ~index] 
    node_list = np.where(~index)[0] + 1
    A_undirected = A_actual + np.transpose(A_actual)
    A_unweighted = np.heaviside(A_undirected, 0)
    M = np.sum(A_actual > 0)
    return A_actual, A_undirected, A_unweighted, node_list, M, A, np.heaviside(A, 0)

def beta_n(network_type, method, snapshot_num, beta_set, n, S, plot, alpha=None, beta=None):
    """TODO: Docstring for change_n.

    :weights: TODO
    :method: TODO
    :n: TODO
    :returns: TODO

    """
    global data_no_loop 
    global index_separate 
    global N_total 
    global month_num 

    if network_type == 'Caviar':
        N_total = 110
        month_num = 11
    else:
        data_no_loop, index_separate, N_total = globals()['load_' + network_type]()
        month_num = np.size(index_separate)

    "directed"
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
            plt.plot(n, success_rate[:, i], linestyle = next(linestyle), linewidth = next(linewidth), alpha = alpha_c, marker = next(marker), label=f'$\\beta=${beta}')
            plt.ylabel('Recall of prediction', fontsize=fs)
        elif plot =='prediction':
            plt.plot(n, prediction[:, i], linestyle = next(linestyle), linewidth = next(linewidth), alpha = alpha_c, marker = next(marker), label=f'$\\beta=${beta}')
            plt.ylabel('Successfully predicted edges', fontsize=fs)
    "unweighted"
    prediction = np.zeros((np.size(n)))
    success_rate = np.zeros((np.size(n)))
    for portion, j in zip(n, range(np.size(n))):
        t1 = time.time()
        prediction[j], success_rate[j] = predictor(snapshot_num, 'unweighted', method, portion, S)
        t2 = time.time()
        print('unweighted', portion, t2 - t1)
    if plot == 'success':
        plt.plot(n, success_rate, linestyle = next(linestyle), linewidth = next(linewidth), alpha = alpha_c, marker = next(marker), label='unweighted')
        plt.ylabel('Recall of prediction', fontsize=fs)
    elif plot =='prediction':
        plt.plot(n, prediction, linestyle = next(linestyle), linewidth = next(linewidth), alpha = alpha_c, marker = next(marker), label='unweighted')
        plt.ylabel('Successfully predicted edges', fontsize=fs)

    "weighted"
    for portion, j in zip(n, range(np.size(n))):
        t1 = time.time()
        prediction[j], success_rate[j] = predictor(snapshot_num, 'weighted', method, portion, S)
        t2 = time.time()
        print('weighted', portion, t2 - t1)
    if plot == 'success':
        plt.plot(n, success_rate, linestyle = next(linestyle), linewidth = next(linewidth), alpha = alpha_c, marker = next(marker), label='weighted')
        plt.ylabel('Recall of prediction', fontsize=fs)
    elif plot =='prediction':
        plt.plot(n, prediction, linestyle = next(linestyle), linewidth = next(linewidth), alpha = alpha_c, marker = next(marker), label='weighted')

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

def motif(snapshot_num, S, a=0):
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
    A_actual, A_undirected, A_unwt, node_list, M, A, A_full_unweighted = merge_weights(snapshot_num, S)
    N = np.size(A, 1)
    A_polar = np.zeros((N, N))
    A_balance = np.zeros((N, N))
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
            M1.add((i, j, k))

    for i in np.where(bi_num >= 1)[0]:
        for j in np.where(A_bi[i]>0)[0]:
            if np.sum(A_pos[i] > 0) > 0:
                for k in np.where(A_pos[i] > 0)[0]:
                    M2.add((i, j, k))

            if np.sum(A_pos[:, i] > 0) > 0:
                for k in np.where(A_pos[:, i] > 0)[0]:
                    M3.add((i, j, k))

    for i in range(np.size(A, 1)):
        if np.sum(A_pos[i] > 0) > 1:
            index_jk = np.where(A_pos[i]>0)[0]
            jk = list(itertools.combinations(index_jk, 2))
            for j, k in jk:
                M4.add((i, j, k))
        if np.sum(A_pos[:, i] > 0) > 1:
            index_jk = np.where(A_pos[:, i]>0)[0]
            jk = list(itertools.combinations(index_jk, 2))
            for j, k in jk:
                M5.add((i, j, k))
        if np.sum(A_pos[i] > 0) >= 1:
            for j in np.where(A_pos[i]>0)[0]:
                if np.sum(A_pos[j] > 0) > 0:
                    for k in np.where(A_pos[j] > 0)[0]:
                        M6.add((j, i, k))
    "delete duplicate elements"
    M = (M1, M2, M3, M4, M5, M6)
    M_weighted = np.zeros((6, N, N))
    M_unweighted = np.zeros((6, N, N))
    for m, mot_index in zip(M, range(6)):
        for mot in m:
            i, j, k = mot
            M_weighted[mot_index, j, k] += A[i, j] + A[i, k] + A[j, i] + A[k, i]
            M_unweighted[mot_index, j, k] +=1
    M_weighted = M_weighted+ np.transpose(M_weighted, (0, 2, 1))
    M_unweighted = M_unweighted+ np.transpose(M_unweighted, (0, 2, 1))
    return M_weighted, M_unweighted

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

def ml_features(network_type, snapshot_num, S, noise, rand_state):
    """TODO: Docstring for ml_features.

    :network_type: TODO
    :snapshot_num: TODO
    :returns: TODO

    """
    global data_no_loop 
    global index_separate 
    global N_total 
    global month_num 

    if network_type == 'Caviar':
        N_total = 110
        month_num = 11
    else:
        data_no_loop, index_separate, N_total = globals()['load_' + network_type]()
        month_num = np.size(index_separate)
    A_actual, A_undirected, A_unweighted, node_list, M, A, _ = merge_weights(snapshot_num, S)
    if noise > 0:
        r = np.random.RandomState(rand_state)
        A_noise = A + r.randint(-1 * noise, noise, size=(N_total, N_total))
        A_noise[A==0] = 0
        A_noise[A_noise<0] = 0
        A = A_noise

    A_label = np.heaviside(A , 0)
    A_label = np.heaviside(A + A.transpose(), 0)

    CN = np.dot(np.heaviside(A, 0), np.heaviside(A, 0))
    M_weighted, M_unweighted = motif(snapshot_num, S)
    A_out = np.zeros((N_total, N_total))
    A_in = np.zeros((N_total, N_total))
    A_CN = np.zeros((N_total, N_total))
    for  i in range(N_total):
        A_out[i] = np.sum((A[i] + A) * np.heaviside(A[i]*A, 0), 1)
        A_in[i] = np.sum((A[:, i] + A.transpose()) * np.heaviside(A[:, i]*A.transpose(), 0), 1)
        AAT = A + A.transpose()
        A_CN[i] = np.sum((AAT[i] + AAT) * np.heaviside(AAT[i]*AAT, 0), 1)

    G = nx.from_numpy_matrix(A_label)
    closeness_dict = nx.closeness_centrality(G)
    closeness = np.array(list(closeness_dict.values()))
    closeness_i = np.broadcast_to(closeness, (N_total, N_total))
    closeness_j = np.broadcast_to(closeness.reshape(N_total, 1), (N_total, N_total))

    d_out = np.sum(A, 1)
    d_in = np.sum(A, 0)

    d_out = np.sum(A>0, 1)
    d_in = np.sum(A>0, 0)

    Di_out = np.broadcast_to(d_out, (N_total, N_total))
    Dj_out = np.broadcast_to(d_out.reshape(N_total, 1), (N_total, N_total))
    Di_in = np.broadcast_to(d_in, (N_total, N_total))
    Dj_in = np.broadcast_to(d_in.reshape(N_total, 1), (N_total, N_total))
    D = Di_in + Di_out
    feature = np.ravel(A_CN).reshape(np.size(np.ravel(A_CN)), 1)
    feature = np.vstack((M_weighted.reshape(6, N_total*N_total))).transpose()
    feature = np.vstack((np.ravel(Di_out), np.ravel(Dj_out))).transpose()
    feature = np.vstack((np.ravel(Di_out), np.ravel(Dj_out), np.ravel(Di_in), np.ravel(Dj_in))).transpose()
    feature = np.vstack((np.ravel(A_CN), np.ravel(A_out), np.ravel(A_in), np.ravel(Di_out), np.ravel(Dj_out), np.ravel(Di_in), np.ravel(Dj_in))).transpose()
    feature = np.vstack((np.ravel(A_out), np.ravel(A_in), np.ravel(Di_out), np.ravel(Dj_out), np.ravel(Di_in), np.ravel(Dj_in))).transpose()
    return feature, np.ravel(A_label)

def ml_logistic_three(network_type, snapshot_num, S, k, test, noise, rand_state):
    """TODO: Docstring for ml_logistic.

    :network_type: TODO
    :snapshot_num: TODO
    :S: TODO
    :returns: TODO

    """
    interval = 1
    feature1, A1_noise = ml_features(network_type, snapshot_num, S, noise, rand_state)
    feature2, A2_noise = ml_features(network_type, snapshot_num + 1 * interval, S, noise, rand_state)
    feature2, A2 = ml_features(network_type, snapshot_num + 1 * interval, S, 0, 0)
    feature3, A3 = ml_features(network_type, snapshot_num + 5 * interval, S, 0, 0)

    Y1_train = np.where(A2_noise - A1_noise == 1)[0]
    Y1_train_num = np.size(Y1_train)
    Y0_train = np.where(A2_noise == 0)[0]
    Y0_train_num = np.size(Y0_train)
    X_train = np.vstack((feature1[Y1_train], feature1[Y0_train]))
    Y_train = np.hstack((np.ones(Y1_train_num), np.zeros(Y0_train_num)))

    Y1_test = np.where(A3-A2  == 1)[0]
    Y0_test = np.where(A3 == 0)[0]
    X_test = np.vstack((feature2[Y1_test], feature2[Y0_test]))
    Y_test = np.hstack((np.ones(np.size(Y1_test)), np.zeros(np.size(Y0_test))))

    balance_k = Y0_train_num/Y1_train_num

    clf = LogisticRegression(class_weight={0:1, 1:k}, random_state=0, solver='lbfgs').fit(X_train, Y_train)
    #clf = svm.SVC(class_weight={0:1, 1:k}, probability=True).fit(X_train, Y_train)
    if test == 'test':
        X_test = X_test
        Y_test = Y_test
    elif test == 'train':
        X_test = X_train
        Y_test = Y_train
        

    truth = Y_test
    probability = clf.predict_proba(X_test[:, :])
    prediction = clf.predict(X_test[:, :])  # default threshold is 0.5
    #threshold = 0.5
    #prediction = (probability[:, 1] >= threshold).astype(bool)
    tru_pos = np.where(Y_test == 1)[0]
    pre_pos = np.where(prediction == 1)[0]
    common = np.array(list(set(tru_pos).intersection(pre_pos))) 
    tp = np.size(common)
    fp = np.size(pre_pos) - np.size(common)
    fn = np.size(tru_pos) - np.size(common)
    tn = np.size(prediction) - tp - fp - fn

    accuracy = (tp + tn)/(tp+ tn + fp + fn)
    precision = (tp)/ (tp + fp)
    recall = tp/ (tp + fn)
    fscore = 2 * (precision * recall)/ (precision + recall)

    return truth, prediction, probability, accuracy, precision, recall, fscore, balance_k

def ml_logistic_two(network_type, snapshot_num, S, k, test):
    """TODO: Docstring for ml_logistic.

    :network_type: TODO
    :snapshot_num: TODO
    :S: TODO
    :returns: TODO

    """
    feature1, A1 = ml_features(network_type, snapshot_num, S)
    feature2, A2 = ml_features(network_type, snapshot_num + 1, S)

    Y1_train = np.where(A1 == 1)[0]
    Y1_train_num = np.size(Y1_train)
    Y0_train = np.where(A1 == 0)[0]
    Y0_train_num = np.size(Y0_train)
    X_train = np.vstack((feature1[Y1_train], feature1[Y0_train]))
    Y_train = np.hstack((np.ones(Y1_train_num), np.zeros(Y0_train_num)))

    Y1_test = np.where(A2 == 1)[0]
    Y0_test = np.where(A2 == 0)[0]
    X_test = np.vstack((feature2[Y1_test], feature2[Y0_test]))
    Y_test = np.hstack((np.ones(np.size(Y1_test)), np.zeros(np.size(Y0_test))))

    balance_k = Y0_train_num/Y1_train_num
    clf = LogisticRegression(class_weight={0:1, 1:k}, random_state=0, solver='lbfgs').fit(X_train, Y_train)
    # clf = svm.SVC(class_weight={0:1, 1:k}, probability=True).fit(X_train, Y_train)
    if test == 'test':
        X_test = X_test
        Y_test = Y_test
    elif test == 'train':
        X_test = X_train
        Y_test = Y_train
        
    truth = Y_test
    prediction = clf.predict(X_test[:, :])
    probability = clf.predict_proba(X_test[:, :])

    tru_pos = np.where(Y_test == 1)[0]
    pre_pos = np.where(prediction == 1)[0]
    common = np.array(list(set(tru_pos).intersection(pre_pos))) 
    tp = np.size(common)
    fp = np.size(pre_pos) - np.size(common)
    fn = np.size(tru_pos) - np.size(common)
    tn = np.size(prediction) - tp - fp - fn

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = (tp)/ (tp + fp)
    recall = tp/ (tp + fn)
    fscore = 2 * (precision * recall)/ (precision + recall)

    return truth, prediction, probability, accuracy, precision, recall, fscore, balance_k

def ml_plot_range(network_type, S, plot_range, k=1):
    """TODO: Docstring for ml_plot.

    :network_type: TODO
    :S: TODO
    :returns: TODO

    """
    accuracy = np.zeros(np.size(plot_range))
    precision = np.zeros(np.size(plot_range))
    recall = np.zeros(np.size(plot_range))
    fscore = np.zeros(np.size(plot_range))
    for snapshot_num, i in zip(plot_range, range(np.size(plot_range))):
        print(i)
        truth, predicion, probability, accuracy[i], precision[i], recall[i], fscore[i]= ml_logistic_two(network_type, snapshot_num, S, k)
        fpr, tpr, threshold = skmetrics.roc_curve(truth, probability[:, 1], pos_label=1)
        roc_auc = skmetrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth = 3, alpha = alpha_c, label = f'No.{i+1}'+ '_AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1],'k--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate', fontsize=fs)
        plt.xlabel('False Positive Rate', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='lower right', fontsize=legendsize)
    plt.show()


    plt.plot(plot_range+1, accuracy, linewidth = 3, alpha = alpha_c, marker = next(marker), label='accuracy')
    plt.plot(plot_range+1, precision, linewidth = 3, alpha = alpha_c, marker = next(marker), label='precision')
    plt.plot(plot_range+1, recall, linewidth = 3, alpha = alpha_c, marker = next(marker), label='recall')
    plt.plot(plot_range+1, fscore, linewidth = 3, alpha = alpha_c, marker = next(marker), label='F-score')

    plt.xlabel('Snapshot number', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(fontsize=legendsize)
    plt.show()

def ml_plot_k(network_type, S, snapshot_num, k, noise=0, rand_state=0):
    """TODO: Docstring for ml_plot.

    :network_type: TODO
    :S: TODO
    :returns: TODO

    """
    accuracy = np.zeros(np.size(k))
    precision = np.zeros(np.size(k))
    recall = np.zeros(np.size(k))
    fscore = np.zeros(np.size(k))

    for k_value, i in zip(k, range(np.size(k))):
        truth, predicion, probability, accuracy[i], precision[i], recall[i], fscore[i], balance_k = ml_logistic_three(network_type, snapshot_num, S, k_value, 'train', noise, rand_state)
        # truth, predicion, probability, accuracy[i], precision[i], recall[i], fscore[i], balance_k = ml_logistic_two(network_type, snapshot_num, S, k_value, 'train')
    index = np.argmax(fscore)
    k_max = k[index]
    for k_value, i in zip(k, range(np.size(k))):
        print(k_value)
        # truth, predicion, probability, accuracy[i], precision[i], recall[i], fscore[i], balance_k = ml_logistic_two(network_type, snapshot_num, S, k_value, 'test')
        truth, predicion, probability, accuracy[i], precision[i], recall[i], fscore[i], balance_k = ml_logistic_three(network_type, snapshot_num, S, k_value, 'test', noise, rand_state)
        fpr, tpr, threshold = skmetrics.roc_curve(truth, probability[:, 1], pos_label=1)
        roc_auc = skmetrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth = 3, alpha = alpha_c, label = 'k=' +str(round(k_value, 1))+ '_AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1],'k--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate', fontsize=fs)
        plt.xlabel('False Positive Rate', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='lower right', fontsize=legendsize)
    plt.close()

    index_test = np.argmax(fscore)
    k_max_test = k[index_test]
    plt.plot(k, accuracy, linewidth = 3, alpha = alpha_c, marker = next(marker), label='accuracy')
    plt.plot(k, precision, linewidth = 3, alpha = alpha_c, marker = next(marker), label='precision')
    plt.plot(k, recall, linewidth = 3, alpha = alpha_c, marker = next(marker), label='recall')
    plt.plot(k, fscore, linewidth = 3, alpha = alpha_c, marker = next(marker), label='Fscore')

    ''''
    truth, predicion, probability, accuracy, precision, recall, fscore, balance_k = ml_logistic_two(network_type, snapshot_num, S, balance_k, 'test')
    plt.plot(balance_k, fscore, 'o', color='k')
    '''

    plt.plot(k_max, fscore[index], 'o', color='g', markersize=9)

    plt.plot(k_max_test, fscore[index_test], 'o', color='k', markersize=9)


    plt.xlabel('factor $k$ ', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='best', fontsize=legendsize)
    plt.savefig(f'../report/caviar_ml_logistic_snap{snapshot_num}_small_performance_k.png')
    #plt.show()

def ml_random(network_type, S, snapshot_num, k, noise, realization, selection_num):
    """TODO: Docstring for ml_plot.

    :network_type: TODO
    :S: TODO
    :returns: TODO

    """
    accuracy = np.zeros(np.size(k))
    precision = np.zeros(np.size(k))
    recall = np.zeros(np.size(k))
    fscore = np.zeros(np.size(k))

    f_score_set = np.zeros((np.size(k), selection_num))
    recall_set = np.zeros((np.size(k), selection_num))
    for k_value, i in zip(k, range(np.size(k))):
        truth, prediction, probability, _, _, _, _, _ = ml_logistic_three(network_type, snapshot_num, S, k_value, 'test', 0, 0)
        print(k_value)
        prediction_ensemble = np.zeros((realization, np.size(prediction)))
        for j in range(realization):
            rand_state = j
            truth, prediction_ensemble[j], probability, _, _, _, _, _ = ml_logistic_three(network_type, snapshot_num, S, k_value, 'test', noise, rand_state)
        prediction_sum = np.sum(prediction_ensemble, 0)
        tru_index = np.where(truth == 1)[0]
        pred_top = np.argsort(prediction_sum)[::-1]
        min_select = np.sum(prediction_sum == realization)
        print(min_select)
        selection = np.linspace(max(5, min_select), min(3* min_select, np.size(pred_top)), selection_num, dtype=int)
        for select_num, s in zip(selection, range(np.size(selection))):
            pred_select = pred_top[:select_num]
            common = np.array(list(set(tru_index).intersection(pred_select))) 
            tp = np.size(common)
            fp = np.size(pred_select) - np.size(common)
            fn = np.size(tru_index) - np.size(common)
            tn = np.size(prediction) - tp - fp - fn

            print(tp)
            accuracy = (tp + tn)/(tp+ tn + fp + fn)
            precision = (tp)/ (tp + fp)
            recall = tp/ (tp + fn)
            recall_set[i, s] = recall
            f_score_set[i, s] = 2 * (precision * recall)/ (precision + recall)

        plt.plot(selection, f_score_set[i], linewidth = 3, alpha = alpha_c, marker = next(marker), label='k=' + str(k_value) )
        #plt.plot(selection, recall_set[i], linewidth = 3, alpha = alpha_c, marker = next(marker), label='k=' + str(k_value) + 'recall')

    plt.xlabel('factor $k$ ', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='best', fontsize=legendsize)
    plt.savefig(f'../report/caviar_ml_logistic_snap{snapshot_num}_small_performance_random{noise}.png')
    #plt.show()

def statistic_ana(network_type, snapshot_range, S, feature_index):
    """TODO: Docstring for statistic_ana.

    :network_type: TODO
    :: TODO
    :returns: TODO

    """
    feature_pos = []
    feature_neg = []
    feature_min = 0
    feature_max = 0

    for i in snapshot_range:
        feature1, label1 = ml_features(network_type, i, S)
        index_neg = np.where(label1 == 0)[0]
        index_pos = np.where(label1 == 1)[0]
        feature_pos = np.append(feature_pos, feature1[index_pos, feature_index])
        feature_neg = np.append(feature_neg, feature1[index_neg, feature_index])
        feature_min = min(feature_min, np.min(feature1[:, feature_index]))
        feature_max = max(feature_max, np.max(feature1[:, feature_index]))
    pos_dis, bins = np.histogram(feature_pos, bins = np.arange(feature_min, feature_max+1, 1), density=True)
    neg_dis, bins = np.histogram(feature_neg, bins = np.arange(feature_min, feature_max+1, 1), density=True)
    pos_dis = pos_dis * 1
    neg_dis = neg_dis * 1
    plot_index = max(np.where(neg_dis>1e-3)[0][-1], np.where(pos_dis>1e-3)[0][-1])
    plt.plot(bins[:plot_index], pos_dis[:plot_index], linewidth = 3, alpha = alpha_c, label='Positive')
    plt.plot(bins[:plot_index], neg_dis[:plot_index], linewidth = 3, alpha = alpha_c, label='negative')
    plt.xlabel('Feature variable', fontsize=fs)
    plt.ylabel('Occurrence frequency', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='upper right', fontsize=legendsize)
    plt.show()
    return pos_dis, neg_dis

def new_edge_feature(network_type, S, snapshot_num):
    interval = 1
    feature1, A1 = ml_features(network_type, snapshot_num, S, 0, 0)
    feature2, A2 = ml_features(network_type, snapshot_num + 1 * interval, S, 0, 0)
    new_edges = np.where(A2 - A1 == 1)[0]
    non_edges = np.where(A2 == 0)[0]
    new_feature = feature1[new_edges]
    non_feature = feature1[non_edges]
    return new_feature, non_feature

   
    
ticksize = 15
legendsize = 12
fs = 20 
alpha_c = 0.8

network_type = 'ia_call'
network_type = 'message'
network_type = 'email'
network_type = 'call'
network_type = 'core_call'
network_type = 'most_call'
network_type = 'Caviar'

weights = 'weighted'
beta = 0
method = 'CN'
plot = 'prediction'
plot = 'success'

n = 0.1
n = np.arange(0.01, 0.21, 0.05)
n = np.arange(0.1, 1.51, 0.1)
beta_set = np.arange(0, 1.1, 0.2)
S = 'inf'
S = 0.5
snapshot_num = 7
realization = 1
plot_range = np.arange(8)
snapshot_range = np.arange(8)
k = np.arange(0.51, 5, 0.1)
k = np.arange(1, 10, 0.5)
k[0] = 1
k = np.arange(1, 10, 1)
feature_index = 4
noise = 0
realization = 1000
selection = np.arange(20, 30, 1)
select_num = 10
# accuracy, precision, recall, fscore = ml_logistic(network_type, snapshot_num, S)
# ml_plot_range(network_type, S, plot_range)
# ml_plot_k(network_type, S, snapshot_num, k)
for snapshot_num in range(1, 2):
    #ml_random(network_type, S, snapshot_num, k, noise, realization, select_num)
    ml_plot_k(network_type, S, snapshot_num, k)
    #new_feature, non_feature = new_edge_feature(network_type, S, snapshot_num)

#feature_pos, feature_neg = statistic_ana(network_type, snapshot_range, S, feature_index)

# prediction, success = beta_n(network_type, method, snapshot_num, beta_set, n, S, plot, alpha=None, beta=None)


'''
count_all, count_percentage, with_cn, M_num_all = new_edge_pattern(S)

plot_range = np.arange(7)
plot_type = 'percent'
plot_motif(count_all, count_percentage, with_cn, M_num_all, plot_range, plot_type)
plt.show()
'''
