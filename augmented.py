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
from numpy import linalg as LA  

marker = itertools.cycle(('d', 'v', 'o', '*'))
linestyle = itertools.cycle(('-', '-', '-.', '-.', '--', '--', ':', ':')) 
linestyle = itertools.cycle(('-', '--')) 
linewidth = itertools.cycle((3.5, 2, 3.5, 2, 3.5, 2, 3.5, 2)) 


mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:red', 'tab:orange', 'tab:cyan','tab:pink', 'tab:purple', 'tab:green',  'tab:brown', 'tab:olive',  'tab:grey','black']) * cycler(linestyle = ['--', '-']))
color=itertools.cycle(('tab:blue', 'tab:red', 'tab:orange', 'tab:cyan','tab:pink', 'tab:purple', 'tab:green',  'tab:brown', 'tab:olive',  'tab:grey','black'))


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
    total_num = np.size(np.unique(month))
    index_separate = [np.where(month == i)[0][-1]+1 for i in range(np.min(np.unique(month)), np.max(np.unique(month)) +1 )]

    node_list = []
    neighbor_list = []
    pair_list = []
    for i, j in zip(np.append(0, index_separate[:-1]), index_separate):
        neighbor_dict = {}
        pairs = []
        data_month = data_no_loop[:, i:j]
        nodes = np.unique(data_month[1:])
        node_list.append(nodes)
        node1 = data_month[1]
        node2 = data_month[2]
        for n1, n2 in zip(node1, node2):
            if n1 in neighbor_dict.keys():
                neighbor_dict[n1].extend([n2])
            else:
                neighbor_dict[n1] = [n2]
            if n2 in neighbor_dict.keys():
                neighbor_dict[n2].extend([n1])
            else:
                neighbor_dict[n2] = [n1]

            if n1<n2:
                pairs.append([n1, n2])
            elif n1>n2:
                pairs.append([n2, n1])
        neighbor_list.append(neighbor_dict)
        pair_set = set([tuple(i) for i in pairs])
        pair_list.append(pair_set)
    return total_num, node_list, neighbor_list, pair_list


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

def load_caviar():
    """TODO: Docstring for snapshot_network.

    :snapshot_month: TODO
    :returns: TODO

    """

    total_num = 11
    node_list = []
    neighbor_list = []
    pair_list = [] 
    for  i in range(total_num):
        data = np.array(pd.read_csv(f'../Datasets Caviar/CAVIAR{i}.csv', header=None).iloc[:, :])
        nodes = data[0, 1:].astype(int)
        node_list.append(nodes)
        N = np.size(nodes)
        A = data[1:, 1:]
        A_undirected = A + np.transpose(A)
        neighbor_dict = {}
        pairs = []
        node1, node2 = np.nonzero(A_undirected)
        for n1, n2 in zip(node1, node2):
            w = int(A_undirected[n1, n2])
            n1 = nodes[n1]
            n2 = nodes[n2]
            if n1 in neighbor_dict.keys():
                neighbor_dict[n1].extend([n2]*w)
            else:
                neighbor_dict[n1] = [n2]*w
            if n1<n2:
                pairs.append([n1, n2])
            elif n1>n2:
                pairs.append([n2, n1])
        neighbor_list.append(neighbor_dict)
        pair_set = set([tuple(i) for i in pairs])
        pair_list.append(pair_set)

    return total_num, node_list, neighbor_list, pair_list

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

def load_coauthor(dataset, co):
    """TODO: Docstring for load_coauthor.
    :returns: TODO

    """
    coauthorship = '../coauth-' + dataset + '-full/'
    #node_label = np.array(pd.read_csv(coauthorship + 'coauth-DBLP-full-node-labels.txt', header=None).iloc[:, 0])
    node_nverts = np.array(pd.read_csv(coauthorship + 'coauth-' + dataset + '-full-nverts.txt', header=None).iloc[:, 0])
    simplex = np.array(pd.read_csv(coauthorship + 'coauth-' + dataset + '-full-simplices.txt', header=None).iloc[:, 0])
    timestamp = np.array(pd.read_csv(coauthorship + 'coauth-' + dataset + '-full-times.txt', header=None).iloc[:, 0])
    node_simplex = np.array(np.split(simplex, np.cumsum(node_nverts))[:-1], dtype=object)

    if co == 1:
        noalone_index = np.where(node_nverts > 1)[0]
        node_nverts_co = node_nverts[noalone_index]
        node_simplex_co = node_simplex[noalone_index]
        timestamp_co = timestamp[noalone_index]
    time_start = np.min(timestamp_co)
    time_end = np.max(timestamp_co)
    time_order = []
    time_length = []
    for i in range(time_start, time_end+1, 1):
        time_index = np.where(timestamp_co== i)[0]
        time_length.append(np.size(time_index))
        time_order.extend(time_index)
    node_nverts_order = np.split(node_nverts_co[time_order], np.cumsum(time_length))[:-1] 
    node_simplex_order = np.split(node_simplex_co[time_order], np.cumsum(time_length))[:-1] 
    return time_start, time_end, node_nverts_order, node_simplex_order

def terrorist(dataset):
    """TODO: Docstring for terrorism.
    :returns: TODO

    """
    #data = pd.read_csv('../Bali2_Relations_Public_Version2.csv', header=None)
    if dataset == 'CE':
        data = pd.read_csv('../CE_Relations_Public_Version2.csv', header=None)
        tie_year = list(data.iloc[1:, 4])
        node1 = list(data.iloc[1:, 1])
        node2 = list(data.iloc[1:, 3])

    elif dataset == 'AE':
        data = pd.read_csv('../AE_Relations_Public_Version2.csv', header=None)
        tie_year = list(data.iloc[1:, 4])
        node1 = list(data.iloc[1:, 2])
        node2 = list(data.iloc[1:, 3])

    elif dataset == 'Bali1':
        data = pd.read_csv('../Bali1_Relations_Public_Version2.csv', header=None)
        tie_year = list(data.iloc[1:, 4])
        node1 = list(data.iloc[1:, 2])
        node2 = list(data.iloc[1:, 3])


    year = []
    repeat_index = []
    for i, j  in zip(tie_year, range(len(tie_year))):
        if pd.isnull(i) == False and len(i) == 4:
            year.append(i)
        else:
            repeat_index.append(j)
            #operator.extend(i.split(';'))
    single_index = np.setdiff1d(np.arange(len(tie_year)), repeat_index).tolist()
    year = np.array(list(map(int, year)))
    node1 = np.array(list(map(int, np.array(node1)[single_index].tolist())))
    node2 = np.array(list(map(int, np.array(node2)[single_index].tolist())))
    node_simplex = np.vstack((year, node1, node2))
    node_simplex_order = node_simplex[:, year.argsort()]
    year_index_order = []
    year_range = np.sort(np.unique(year))
    for i in year_range:
        year_index_order.append(np.where(node_simplex_order[0]==i)[0][0])
    year_index_order.append(np.size(year))

    node_simplex_collection = []
    for i, j in zip(year_index_order[:-1], year_index_order[1:]): 
        node_simplex_collection.append(node_simplex_order[1:, i:j].transpose())

    return node_simplex_collection

def coauthor_neighbor(connection):
    """TODO: Docstring for coauthor_exist_edges.

    :arg1: TODO
    :returns: TODO

    """
    neighbor_dict = {}
    for pair in connection:
        for node in pair:
            neighbor = np.setdiff1d(pair, node)
            if node in neighbor_dict.keys():
                neighbor_dict[node].extend(list(neighbor))
            else:
                neighbor_dict[node] = list(neighbor)
    return neighbor_dict

def coauthor_exist_edges(connection):
    """TODO: Docstring for coauthor_exist_edges.

    :arg1: TODO
    :returns: TODO

    """

    pair_list = []
    for pair in connection:
        for i in set(itertools.combinations(pair, 2)):
            if i[0] < i[1]:
                pair_list.append(i)
            elif i[0] > i[1]:
                pair_list.append(i[::-1])  
    pair_list = np.array(pair_list)
    "convert to set format to make the element unique"
    pair_set = set([tuple(i) for i in pair_list])
    return pair_set

def node_age(node_list):
    """TODO: Docstring for node_age.

    :node: TODO
    :returns: TODO

    """
    age = []
    node_set = []
    for node_i, i in zip(node_list, range(len(node_list))):
        node_i = np.unique(node_i)
        for node in node_i:
            if node not in node_set:
                node_set.append(node)
                age.append(i)
    mapping = dict(zip(node_set, age))
    return mapping

def coauthor_CN(index, mapping, neighbor_list, pair_set, weights, age_effect):
    """TODO: Docstring for coauthor_oneshot.

    :arg1: TODO
    :returns: TODO

    """
    "node pair 1st < 2nd; no repetition"
    CN_all = set()
    neighbor_weighted = {}
    neighbor_only = {}
    for neighbor_dict in neighbor_list:
        node_list = np.array(list(neighbor_dict.keys()))
        for node in node_list:
            neighbors = collections.Counter(neighbor_dict[node])
            key = neighbors.keys()
            value = np.array(list(neighbors.values()))
            node_neighbor = {}
            for i, j in zip(key, value):
                node_neighbor[i] = j
            if node in neighbor_weighted.keys():
                node_neighbor = dict(collections.Counter(neighbor_weighted[node]) + collections.Counter(node_neighbor))
            neighbor_weighted[node] = node_neighbor
            neighbor_only[node] = list(node_neighbor.keys())
    CN_all = set()
    node_all = list(neighbor_only.keys())
    for node in node_all:
        neighbor = neighbor_only[node]
        if np.size(neighbor) > 1:
            for i, j in set(itertools.combinations(neighbor, 2)):
                if i < j:
                    CN_all.add((i, j))
                else:
                    CN_all.add((j, i))

    CN_set = pair_set.union(CN_all) - pair_set
    CN_info, top_edge = CN_value(index, mapping, CN_set, node_all, neighbor_only, neighbor_weighted)
    CN_exist_info, top_exsit_edge = CN_value(index, mapping, pair_set, node_all, neighbor_only, neighbor_weighted)
    return CN_info, top_edge, CN_exist_info, top_exsit_edge

def CN_value(index, mapping, CN_set, node_all, neighbor_only, neighbor_weighted):
    """TODO: Docstring for CN_value.

    :arg1: TODO
    :returns: TODO

    """
    CN_dict = {}
    for node in node_all:
        neighbor = neighbor_only[node]
        if np.size(neighbor) > 1:
            for i, j in set(itertools.combinations(neighbor, 2)):
                if i > j:
                    k = j
                    j = i
                    i = k
                if (i, j) in CN_set:
                    neighbor_count = [neighbor_weighted[node][i] + neighbor_weighted[node][j]]

                    if (i, j) in CN_dict.keys():
                        CN_dict[i, j].extend(neighbor_count)
                    else:
                        CN_dict[i, j] = neighbor_count

    CN_values = list(CN_dict.values())
    weighted = np.array([np.sum(i) for i in CN_values])
    unweighted = np.array([np.size(i) for i in CN_values])
    edges = np.array(list(CN_dict.keys()))
    node1 = edges[:, 0]
    node2 = edges[:, 1]

    if age_effect:
        a = 1
        age1 = index - np.array([mapping[int(i)] for i in node1])
        age2 = index - np.array([mapping[int(i)] for i in node2])
        weighted_age = weighted * np.log(age1 +a) * np.log(age2 +a)
        unweighted_age = unweighted * np.log(age1 +a) * np.log(age2 +a)
    else:
        weighted_age = weighted
        unweighted_age = unweighted

    CN_info = np.vstack((node1, node2, weighted_age, unweighted_age)).transpose()
    if weights == 'weighted':
        sort_order = np.argsort(CN_info[:, 2])[::-1]
    elif weights == 'unweighted':
        sort_order = np.argsort(CN_info[:, 3])[::-1]
    top_edge = CN_info[sort_order, :2]
    CN_sort = CN_info[sort_order]

    return CN_sort, top_edge

def original_construct(dataset, co):
    """TODO: Docstring for intermediate.

    :network_type: TODO
    :snapshot_num: TODO
    :method: TODO
    :weights: TODO
    :returns: TODO

    """
    if dataset == 'AE' or dataset == 'CE' or dataset == 'Bali1':
        node_simplex_collection = terrorist(dataset)
        node_simplex_select = node_simplex_collection
        total_num = len(node_simplex_select)

    elif dataset == 'DBLP':
        time_start, time_end, node_nverts_order, node_simplex_order = load_coauthor(dataset, co)
        node_simplex_select = node_simplex_order[time_select[0]-time_start:time_select[1]-time_start + 1]
        total_num = time_select[1] - time_select[0] +1


    neighbor_list = []
    dt_list = []
    pair_list = []
    node_list = []

    for connection in node_simplex_select: 
        neighbor_dict = coauthor_neighbor(connection)
        pair_i = coauthor_exist_edges(connection)
        # neighbor list grouped by timestamp.
        neighbor_list.append(neighbor_dict)
        pair_list.append(pair_i)
        node_i = list(neighbor_dict.keys())
        node_list.append(node_i)

    return total_num, node_list, neighbor_list, pair_list

def augmented(index, mapping, neighbor_index, pair_set, intermediate_num, weights, age_effect):
    """TODO: Docstring for augmented.

    :arg1: TODO
    :returns: TODO

    """

    CN_info, top_edge, _, _ = coauthor_CN(index, mapping, neighbor_index, pair_set, weights, age_effect)

    "intermediate network"
    intermediate_pair_set = pair_set.union(set([tuple(i) for i in top_edge[:intermediate_num]]))
    intermediate_neighbor_dict =  {}
    for i, j in top_edge[:intermediate_num]:
        if i in intermediate_neighbor_dict.keys():
            intermediate_neighbor_dict[i].extend([j])
        else:
            intermediate_neighbor_dict[i] = [j]
        if j in intermediate_neighbor_dict.keys():
            intermediate_neighbor_dict[j].extend([i])
        else:
            intermediate_neighbor_dict[j] = [i]
    intermediate_neighbor_list = np.append(neighbor_index, intermediate_neighbor_dict)

    intermediate_CN_info, intermediate_top_edge, _, _ = coauthor_CN(index, mapping, intermediate_neighbor_list, pair_set, 'unweighted', age_effect)

    CN_dict = {}
    for i in intermediate_CN_info:
        CN_dict[tuple(i[:2].astype(int))] = i[2:].tolist()
    CN_pair = set(CN_dict.keys())
    return CN_dict, CN_pair, intermediate_pair_set, intermediate_neighbor_list 

def AUC(pair_set_future, pair_set, node_index, CN_dict, CN_pair, interval, auc_method):
    """TODO: Docstring for AUC.

    :arg1: TODO
    :returns: TODO

    """
    "To see the new edges in the future "
    pair_new = pair_set_future - pair_set
    " pairs of nodes are grouped according to node index. e.g. [0: interval), [interval: interval*2), ..."
    pair_sort = np.array(list(pair_new))[np.argsort(np.array(list(pair_new))[:, 0])]
    node_sort = np.sort(np.array(node_index), axis=0)
    bound = np.hstack((np.arange(interval, np.max(node_index), interval), np.max(node_index)))
    pair_index = []
    for i in bound:
        if np.sum(pair_sort<=i):
            pair_index.append(np.where(pair_sort<=i)[0][-1])
    "only consider the edges where nodes are already in the current snapshot. "
    newedge_oldnode = []
    for i, j in zip(np.append(0, pair_index[:-1]), pair_index):
        i = int(i)
        j = int(j)
        group_i = pair_sort[i: j] 
        if len(group_i):

            k = np.where(node_sort>=np.min(group_i))[0][0]
            l = np.where(node_sort<=np.max(group_i))[0][-1]
            node_i = np.intersect1d(node_sort[k: l+1], np.unique(group_i))
            newedge_oldnode.extend(group_i[np.where(np.sum(np.sum(group_i.reshape(group_i.shape + (1,)) - node_i == 0, 2), 1) == 2)])
    newedge_oldnode = np.array(newedge_oldnode)

    "new-- score of new edges; new_cn-- new edges "
    new = []
    non = []
    new_cn = []
    for i in newedge_oldnode:
        i = tuple(i)
        if i in CN_pair:
            new.append(CN_dict[i])
            new_cn.append(i)
            
    non_cn = CN_pair - set(new_cn)
    num_new_cn = len(new)
    num_new_nocn = len(newedge_oldnode) - num_new_cn
    for i in non_cn:
        non.append(CN_dict[i])
    num_non_cn = len(non)
    num_old_edge = len(pair_set)
    num_non_nocn = int(len(node_index) * (len(node_index) - 1)/2) - num_new_cn - num_new_nocn - num_non_cn - num_old_edge

    "Only consider the edges with CN"
    if auc_method == 'BKS':
        p_new = np.ones(num_new_cn)/ num_new_cn
        p_non = np.ones(num_non_cn)/num_non_cn
        new_score = np.array(new)
        non_score = np.array(non)

    else:
        p_new = np.hstack((np.ones(num_new_cn), num_new_nocn))/(num_new_cn + num_new_nocn)
        p_non = np.hstack((np.ones(num_non_cn), num_non_nocn))/(num_non_cn + num_non_nocn)
        new_score = np.vstack((np.array(new), np.array([0,0])))
        non_score = np.vstack((np.array(non), np.array([0,0])))

    realization = 1000000
    new_choose = np.random.choice(np.array(new_score)[:, 1], realization, p = p_new)
    non_choose = np.random.choice(np.array(non_score)[:, 1], realization, p = p_non)
    auc = (np.sum(new_choose - non_choose>0) + 0.5*np.sum(new_choose == non_choose))/realization
    return auc

def intermediate_auc(dataset, co, index_range, weights, percentage_range, interval, auc_method, age_effect, cutoff):
    """TODO: Docstring for intermediate_auc.

    :arg1: TODO
    :returns: TODO

    """
    auc = np.zeros((np.size(index_range), np.size(percentage_range)))
    if dataset == 'AE' or dataset == 'CE' or dataset == 'Bali1' or dataset == 'DBLP':
        total_num, node_list, neighbor_list, pair_list = original_construct(dataset, co)
    elif dataset == 'Caviar':
        total_num, node_list, neighbor_list, pair_list = load_caviar()
    elif dataset == 'email':
        total_num, node_list, neighbor_list, pair_list = load_email()

    mapping = node_age(node_list)
    for index, i in zip(index_range, range(np.size(index_range))):
        neighbor_index = neighbor_list[:index+1]
        pair_index = pair_list[:index+1]
        node_index = np.unique(np.concatenate(node_list[:index+1]))
        pair_set = set()
        for pair_i in pair_index:
            pair_set = pair_i.union(pair_set)

        future_index = min(index+1+3, total_num)
        future_index = total_num
        neighbor_future = neighbor_list[:future_index]
        pair_future = pair_list[:future_index]
        pair_set_future = set()
        for pair_i in pair_future:
            pair_set_future = pair_i.union(pair_set_future)

        CN_info, top_edge, CN_exist_info, top_exsit_edge = coauthor_CN(index, mapping, neighbor_index, pair_set, weights, age_effect)
        if cutoff:
            if weights == 'weighted':
                cut_off= np.min(CN_exist_info[:, 2])
                cutoff_index = np.where(CN_info[:, 2]>=cut_off)[0][-1]
            if weights == 'unweighted':
                cut_off= np.min(CN_exist_info[:, 3])
                cutoff_index = np.where(CN_info[:, 3]>=cut_off)[0][-1]
            top_edge = top_edge[:cutoff_index+1]
        
        intermediate_range = np.array(len(top_edge) * percentage_range, int)
        for intermediate_num, j in zip(intermediate_range, range(np.size(intermediate_range))):
            t1 = time.time()
            CN_dict, CN_pair, intermediate_pair_set, intermediate_neighbor_list = augmented(index, mapping, neighbor_index, pair_set, intermediate_num, weights, age_effect)
            auc[i, j] = AUC(pair_set_future, pair_set, node_index, CN_dict, CN_pair, interval, auc_method)
            t2 = time.time()
            print(i, j, t2-t1)
    plot_order = np.argsort(auc[:, 0])[::-1]

    fig, ax = plt.subplots()
    for i in range(np.size(index_range)):
        auc_i = auc[plot_order[i]] 
        index_max = np.argmax(auc_i)
        plt.plot(percentage_range[:index_max+1], auc_i[: index_max+1],  alpha=alpha_c)
        plt.plot(percentage_range[index_max:], auc_i[index_max:],  alpha = alpha_c, label=f'{plot_order[i]+1}')
        plt.plot(0, auc_i[0], 'o' , markersize=4, color=next(color))

    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('AUC', fontsize=fs)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    plt.subplots_adjust(left=0.15, right=0.85, wspace=0.25, hspace=0.25, bottom=0.20, top=0.95)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(fontsize=legendsize, loc='upper left', bbox_to_anchor=(-0.22, 1.0), framealpha=0)
    plt.locator_params(axis='x', nbins=5)
    #save_des = '../manuscript/manuscript090820/figure/'+ dataset + weights 
    #plt.savefig(save_des+ '.svg', format="svg") 
    #plt.savefig(save_des + '.png') 


    #plt.close('all')
    #plt.show()
    return auc

def ane(dataset, co, index_range, weights, percentage_range, interval, auc_method, age_effect, cutoff):
    """TODO: Docstring for ane.

    :arg1: TODO
    :returns: TODO

    """

    weighted_score = np.zeros((np.size(index_range), np.size(percentage_range)))
    unweighted_score = np.zeros((np.size(index_range), np.size(percentage_range)))
    auc_opt = np.zeros((np.size(index_range)))
    auc = np.zeros((np.size(index_range), np.size(percentage_range)))
    if dataset == 'AE' or dataset == 'CE' or dataset == 'Bali1' or dataset == 'DBLP':
        total_num, node_list, neighbor_list, pair_list = original_construct(dataset, co)
    elif dataset == 'Caviar':
        total_num, node_list, neighbor_list, pair_list = load_caviar()
    elif dataset == 'email':
        total_num, node_list, neighbor_list, pair_list = load_email()
    if age_effect: 
        mapping = node_age(node_list)
    else:
        mapping = 0
    for index, i in zip(index_range, range(np.size(index_range))):
        print(i)
        neighbor_index = neighbor_list[:index+1]
        pair_index = pair_list[:index+1]
        node_index = np.unique(np.concatenate(node_list[:index+1]))
        pair_set = set()
        for pair_i in pair_index:
            pair_set = pair_i.union(pair_set)

        future_index = min(index+1+3, total_num)
        future_index = total_num
        neighbor_future = neighbor_list[:future_index]
        pair_future = pair_list[:future_index]
        pair_set_future = set()
        for pair_i in pair_future:
            pair_set_future = pair_i.union(pair_set_future)

        CN_info, top_edge, CN_exist_info, top_exsit_edge = coauthor_CN(index, mapping, neighbor_index, pair_set, weights, age_effect)
        if weights == 'weighted':
            exist_ave = np.mean(CN_exist_info[:, 2], 0)
            possible_num = np.size(np.where(CN_info[:, 2] >exist_ave )[0])
        else:
            exist_ave = np.mean(CN_exist_info[:, 3], 0)
            possible_num = np.size(np.where(CN_info[:, 3] >exist_ave )[0])

        CN_dict, CN_pair, _, _ = augmented(index, mapping, neighbor_index, pair_set, possible_num, weights, age_effect)
        auc_opt[i] = AUC(pair_set_future, pair_set, node_index, CN_dict, CN_pair, interval, auc_method)

        intermediate_range = np.array(len(top_edge) * percentage_range, int)
        for intermediate_num, j in zip(intermediate_range, range(np.size(intermediate_range))):
            t1 = time.time()
            CN_dict, CN_pair, intermediate_pair_set, intermediate_neighbor_list = augmented(index, mapping, neighbor_index, pair_set, intermediate_num, weights, age_effect)

            _, _, CN_inter_info, top_inter_edge = coauthor_CN(index, mapping, intermediate_neighbor_list, intermediate_pair_set, weights, age_effect)
            auc[i, j] = AUC(pair_set_future, pair_set, node_index, CN_dict, CN_pair, interval, auc_method)

            t2 = time.time()
            print(i, j, t2-t1)

    return auc, auc_opt






ticksize = 15
legendsize = 12
fs = 20 
markersize = 8
alpha_c = 0.8
lw = 3

dataset = 'MAG-History'
dataset = 'MAG-Geology'
dataset = 'terrorism'
dataset = 'CE'
co = 1  # the counted paper have more than one author.
index = 2
time_select = [1997, 2006]
S = 0.1
S = 'infinite'
weights = 'weighted'
weights = 'unweighted'
n_set = np.arange(0.01, 0.11, 0.01)
n_set = np.arange(10, 100, 10)

#success_set = coauthor_plot(dataset, co, time_select, weights, n_set, S)
#intermediate_num = 0
#auc = intermediate(dataset, co, index, weights, S, intermediate_num3
intermediate_range = np.arange(0, 20, 1)
dataset = 'Bali1'
index_range = np.arange(8, 17, 1)
dataset = 'email'
dataset = 'Caviar'
index_range = np.arange(0, 10, 1)
dataset = 'DBLP'
index_range = np.arange(0, 9, 1)
percentage_range = np.arange(0, 0.9, 0.1)
interval = 1000
auc_method = 'BKS'
auc_method = 'MC'
age_effect = 0
cutoff = 0
#auc = intermediate_auc(dataset, co, index_range, weights, percentage_range, interval, auc_method, age_effect, cutoff)
auc, auc_opt= ane(dataset, co, index_range, weights, percentage_range, interval, auc_method, age_effect, cutoff)
#auc = AUC(dataset, co, index, weights, S)
#S_auc(dataset, co, index, [S])
'''
S_set  = np.array([0, 0.01, 0.1, 0.2, 0.5, 1, 5, 10])
for index in range(1, 2, 1):
    print(index)
    S_auc(dataset, co, index, S)
'''



