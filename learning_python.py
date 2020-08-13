import numpy as np 
import matplotlib.pyplot as plt 

ticksize = 15
legendsize = 11
fs = 20 

n = 10
ind = np.arange(n)  # the x locations for the groups
width = 0.2       # the width of the bars
node_order = node_total[:n, 0]
node_list = tuple([str(int(x+1)) for x in node_order])
node_norm = node_total[:n, 1]/node_total[:, 1].max()

index_close = [np.where(closeness[:, 0] == node_order[i])[0][0] for i in range(n)] 
closeness_sequence = closeness[index_close]
close_norm = closeness_sequence[:n, 1]/closeness[:, 1].max()


index_degree = [np.where(degree[:, 0] == node_order[i])[0][0] for i in range(n)] 
degree_sequence = degree[index_degree]
degree_norm = degree_sequence[:n, 1]/degree[:, 1].max()

index_between = [np.where(between[:, 0] == node_order[i])[0][0] for i in range(n)] 
between_sequence = between[index_between]
between_norm = between_sequence[:n, 1]/between[:, 1].max()


fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, node_norm, width, color='tab:red')
rects2 = ax.bar(ind+width, close_norm, width, color='tab:blue')
rects3 = ax.bar(ind+width*2, degree_norm, width, color='tab:green')
rects4 = ax.bar(ind+width*3, between_norm, width, color='tab:orange')

label1 = np.arange(n) + 1
label2 = np.array(index_close) + 1
label3 = np.array(index_degree) + 1
label4 = np.array(index_between) + 1

ax.set_xticks(ind+width)
ax.set_xticklabels(node_list)
ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('number of edges', 'closeness centrality', 'degree centrality', 'betweenness_centrality'), fontsize = legendsize )

def autolabel(rects, label):
    for rect, i in zip(rects, range(np.size(rects))):
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.01*h, f'{label[i]}', ha='center', va='bottom')

autolabel(rects1, label1)
autolabel(rects2, label2)
autolabel(rects3, label3)
autolabel(rects4, label4)

plt.xlabel('node sequence', fontsize=fs)
plt.ylabel('normalized value', fontsize=fs)
plt.subplots_adjust(left=0.15, right=0.97, wspace=0.25, hspace=0.25, bottom=0.15, top=0.97)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
# plt.legend(fontsize=legendsize)
#plt.legend(bbox_to_anchor=(0.98, 1.0))
plt.show()

plt.show()


# find the element of array a in array b
np.nonzero(np.in1d(a, b))[0]
