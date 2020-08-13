import numpy as np 
import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
from cycler import cycler
import itertools
import time 
import collections

des_file = '../communication/sorted_work/data/link.npy'
des_file = '../communication/burst/time_series_calls_outgoing.npy'
data = np.load(des_file, allow_pickle=True)
data = pd.read_csv('../communication/time-of-events-outgoing.txt', header=None)
