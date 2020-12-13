import numpy as np
import matplotlib.pyplot as plt

import tqdm 
import tqdm.notebook

from optimization import *

import time
import pickle
# optimal solution
from optimal_solution import *
# from clustering import *
from sklearn.cluster import KMeans


def optimize_with_clusters(g, l, beta, num_clusters=[10, 100],
                           n_iter=20000, mutation_strategy=1, initial_selection_probability=0.0, 
                           precompute_pairwise_dist=False, verbose=True):
    c_n_iter = n_iter // (2*len(num_clusters))
    for i, C in enumerate(num_clusters):
        kmeans = KMeans(n_clusters=C, max_iter=500)
        kmeans.fit(g.x)
        centers = kmeans.cluster_centers_
        population = [sum(g.v[np.where(kmeans.labels_==k)[0]])/len(np.where(kmeans.labels_==k)[0]) for k in range(C)]
        
        class clusters(object):
            def __init__(self):
                self.x = centers
                self.v = population
        clusters = clusters()
#         c_n_iter = n_iter//4
        print(c_n_iter)
        if i==0:
            c_selected_cities=(np.random.rand(len(centers)) <= initial_selection_probability).astype(np.int32)
        else:
            c_selected_cities=np.zeros(len(centers))
            # pick only mega cities from the following set
            init_set = np.unique(kmeans.labels_[np.where(np.in1d(prev_labels, np.nonzero(c_cities_n_convex))==True)[0]])
            # number of cities to pick for initialization
            if str(g)=="G2":
                s = int(1 + np.sqrt(len(init_set))*(2-l)*len(init_set)/len(init_set))
            else:
                s = int(1 + np.sqrt(len(init_set))*(1-l)*len(init_set)/len(init_set))
            c_selected_cities[np.random.choice(init_set, replace=False, size=s)] = 1
        c_cities_n, c_cities_n_convex, c_loss_values, c_loss_value_convex = optimize(clusters, l, beta=beta, 
                                                  n_iter=c_n_iter, mutation_strategy=mutation_strategy, 
                                                  initial_selection_probability=initial_selection_probability,
                                                  precompute_pairwise_dist=False, verbose=True, selected_cities=c_selected_cities)
        prev_labels = kmeans.labels_
    #### 
    # run on all cities
    full_n_iter = n_iter - len(num_clusters) * c_n_iter
    full_selected_cities=np.zeros(len(g.x))
    init_set = np.where(np.in1d(kmeans.labels_, np.nonzero(c_cities_n_convex))==True)[0]
    # number of cities to pick for initialization
    if str(g)=="G2":
        s = int(1 + np.sqrt(len(init_set))*(2-l)*len(init_set)/len(init_set))
    else:
        s = int(1 + np.sqrt(len(init_set))*(1-l)*len(init_set)/len(init_set))
        full_selected_cities[np.random.choice(init_set, replace=False, size=s)] = 1
    cities_n, cities_n_convex, loss_values, loss_value_convex = optimize(g, l, beta=beta, 
                                              n_iter=full_n_iter, mutation_strategy=mutation_strategy, 
                                              initial_selection_probability=initial_selection_probability,
                                              precompute_pairwise_dist=False, verbose=True, selected_cities=full_selected_cities)
    return cities_n, cities_n_convex, loss_values, loss_value_convex