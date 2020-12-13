import matplotlib.pyplot as plt
from matplotlib.path import Path

import numpy as np

import scipy
import scipy as sp
import scipy.stats as st
from scipy.spatial import ConvexHull

import tqdm
import tqdm.notebook

from optimization import *
from optimal_solution import *

import time
import scipy
from scipy import spatial
import itertools

from scipy.cluster.vq import kmeans,vq,whiten


def bruteforce_sol_N(N,N2,l, generator,pairwise_distances):
    # very inefficient solution: O(2^N)
    min_f = np.inf
    all_combination_cities = list(map(list, itertools.product([0, 1], repeat=N)))
    if pairwise_distances is None:
        pairwise_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(generator.x, 'sqeuclidean'))
    for selected_cities in all_combination_cities[1:]:
        f = objective_function_simple(N2, l, generator, selected_cities,
                                      pairwise_distances=pairwise_distances)
        if f < min_f:
            solution = selected_cities
            min_f = f
    return solution, min_f


def plotResult(data,duration,selected_cities_n,selected_cities_n_convex,loss_values,loss_value_convex,num_cities_per_step,verbose=True):
    if verbose:
        print("d= %s seconds" % duration)
        if loss_value_convex is not None:
            print('Final loss '+ str(loss_values[-2])) #be careful, at the end we want to return '-final loss'
            print('Final loss with Convex Hull '+ str(loss_value_convex))
        else:
            print('Final loss '+ str(loss_values[-1]))
    #print(selected_cities_n)

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    #fig.suptitle('Results')
    axes[0].plot(loss_values)
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Iterations')
    axes[0].set_title('Loss Evolution')
    m = selected_cities_n == 1
    axes[1].scatter(data.x[:, 0], data.x[:, 1],label='Non selected Cities')
    axes[1].scatter(data.x[m, 0], data.x[m, 1], c='r',label='Selected cities')
    if selected_cities_n_convex is not None:
        mbis = (selected_cities_n_convex==1) & (selected_cities_n==0)
        mter = (selected_cities_n_convex==0) & (selected_cities_n==1)
        axes[1].scatter(data.x[mbis, 0], data.x[mbis, 1], c='g',label='Added cities (Convex Hull)')
        axes[1].scatter(data.x[mter, 0], data.x[mter, 1], c='y',label='Selected Cities not in Convex Hull')
    axes[1].set_title('Selected cities')
    box = axes[1].get_position()
    axes[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
    
    if num_cities_per_step is not None:
        plt.figure(figsize=(4,2))
        plt.plot(np.arange(n_iter), num_cities_per_step)
        plt.title("#selected cities in each step")
        


def optimize_cluster(N,cities, l, selected_cities_init, betas=[20,100], n_iter=20000, mutation_strategy=0, initial_selection_probability=0.5, precompute_pairwise_dist=False, verbose=True):
    """mutation_strategy = 0: Original mutation proposed by Heloise
       mutation_strategy = 1: Simple strategy which just randomly tries to flip cities
       initial_selection_probability: Probablity at which a city initially is selected (0.5: every city can be selected with 50% chance)

       precompute_pairwise_dist: Enabling this gives slightly better performance, but has quadratic memory complexity
       """

    N2 = cities.x.shape[0]
    selected_cities=selected_cities_init
#     print("done")
    if precompute_pairwise_dist:
        pairwise_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(cities.x, 'sqeuclidean'))
    else:
        pairwise_distances = None

    fs = np.zeros(n_iter)
    all_selected_cities = []
    current_loss_value, max_dist, max_idx, convex_hull = objective_function_(N, l, cities, None, selected_cities, None, pairwise_distances)
    n_iterbeta=n_iter//len(betas)
#     print(n_iterbeta)
    it = tqdm.notebook.tqdm(range(n_iterbeta)) if verbose else range(n_iterbeta)

    state = {
        'selected': selected_cities,
        'loss_value': current_loss_value,
        'max_dist': max_dist,
        'max_idx': max_idx,
        'convex_hull': convex_hull,
        # 'max_points': max_points,
    }
    state['delaunay'] = scipy.spatial.Delaunay(cities.x) if  mutation_strategy == 4 else None
    
    for k in range (len(betas)):
        beta=betas[k]
#         print('beta'+str(beta))
        for m in it:
            fs[m+k*n_iterbeta] = state['loss_value']
            state = step(N, cities, state, beta, l, pairwise_distances,
                mutation_strategy=mutation_strategy)
            all_selected_cities.append(state['selected'])
        all_selected_cities_convex,fs_convex=adding_convex_points(N, l, cities,state)
    return all_selected_cities, all_selected_cities_convex,fs,fs_convex


def do_optimization_cluster(g, l, betas_init=20, beta_last=50, n_iter=10000, mutation_strategy=3, initial_selection_probability=0.5, precompute_pairwise_dist=False, verbose=True, show=False):
    #set number of steps
    N=g.x.shape[0]
    step_cluster=np.int(np.floor(np.log10(N)))
    true_step_cluster=min(3,np.int(np.floor(np.log10(N)))-1)
    print('number of steps= '+str(step_cluster))
    betas=[betas_init]
    totalLoss=[]

    for stepi in range (0,true_step_cluster):
        betas=[betas[0]+10]
        nclusters=N//(10**(step_cluster-1-stepi))
        print('step = '+str(stepi)+', N clusters= '+str(nclusters))
        selected_cities=np.zeros(nclusters)
        is_in_selected_cluster=np.zeros(N)
        if stepi>0:
            #is_in_selected_cluster=[selected_cities_n[clx[k]]==1 for k in range (N)] #not working?
            for k in range(N):
                c=clx[k]
                if selected_cities_n[c]==1:
                    is_in_selected_cluster[k]=1
                    
        data=g.x
        ind = np.argpartition(g.v, -nclusters)[-nclusters:]
        centroids=(data[ind])
        #centroids,_ = kmeans(data,nclusters)
        clx,_ = vq(data,centroids)
        centroids_V=[np.sum(g.v, where=(clx == k)) for k in range (centroids.shape[0])]
        #centroids_V=np.divide(centroids_V,10**(step_cluster-1-stepi))
        g_clusters=G1(N)
        g_clusters.x=np.array(centroids)
        g_clusters.v=np.array(centroids_V)

        if stepi>0 :
            selected_cities=[np.sum((clx==k)*(is_in_selected_cluster))==(sum(clx==k)/2) for k in range (nclusters)]
            selected_cities=np.array(selected_cities)
            for k in range (nclusters):
                in_cluster=clx==k
                is_selected_city=is_in_selected_cluster
                good_cities=in_cluster*is_in_selected_cluster
                if np.sum(good_cities)==(sum(in_cluster)):
                    selected_cities[k]=1

        best_selected_cities_N=np.array([selected_cities[clx[k]] for k in range (N)])
        if show:
            f = objective_function_simple(N, l, g, best_selected_cities_N,pairwise_distances=None)
            print('f of initializing state = '+str(f))
            
        np.random.seed()
        selected_cities_n, selected_cities_n_convex, loss_values,loss_value_convex = optimize_cluster(N,g_clusters, l, selected_cities, betas=betas, 
                                                  n_iter=n_iter,mutation_strategy=mutation_strategy, initial_selection_probability=initial_selection_probability, verbose=False)  
        if type(selected_cities_n) == list:
            selected_cities_n = selected_cities_n[-1]
        else:
            print('problem!')

        if show:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            axes[0].scatter(data[:,0],data[:,1],c=clx ,label='cities')
            sequence = np.arange(centroids.shape[0])
            axes[1].scatter(centroids[:, 0], centroids[:, 1], c=sequence,label='centroids')
            plotResult(g_clusters,0,selected_cities_n,selected_cities_n_convex,loss_values,loss_value_convex,None,verbose=False)
        
        totalLoss=np.concatenate((totalLoss,loss_values))

    print('\n Last step, N= '+str(N))
    is_in_selected_cluster=[selected_cities_n[clx[k]]==1 for k in range (N)]
    selected_cities_n, selected_cities_n_convex, loss_values,loss_value_convex = optimize(g, l, beta=beta_last, n_iter=n_iter,mutation_strategy=mutation_strategy, initial_selection_probability=initial_selection_probability, precompute_pairwise_dist=False,selected_cities=np.array(is_in_selected_cluster), verbose=False)
    if type(selected_cities_n) == list:
        selected_cities_n_result = np.array(selected_cities_n[-1])
    
    totalLoss=np.concatenate((totalLoss,loss_values))
#     if show:  
#         plotResult(g,0,selected_cities_n_convex,selected_cities_n_convex,loss_values,loss_value_convex,None)
#         plt.figure()
#         plt.plot(totalLoss)
#         plt.ylabel('Loss')
#         plt.xlabel('Iterations')
#         plt.title('Loss Evolution')  
    totalLoss=totalLoss[::(true_step_cluster+1)]
    return selected_cities_n_result,selected_cities_n_convex,loss_values,loss_value_convex
    #return selected_cities_n_result,selected_cities_n_convex,totalLoss,loss_value_convex


def do_optimization_cluster_simple(g, l, beta_init=20, n_iter=10000, mutation_strategy=1, initial_selection_probability=0.5, precompute_pairwise_dist=False, verbose=True, show=False):
    #set number of steps
    N=g.x.shape[0]
    step_cluster=np.int(np.floor(np.log10(N)))
    print('number of steps= '+str(step_cluster))
    beta=beta_init
    totalLoss=[]
    ind_old=[]
    best_selected_cities_N=np.zeros(N)

    for stepi in range (0,step_cluster):
        nclusters=N//(10**(step_cluster-1-stepi))
        print('step = '+str(stepi)+', N clusters= '+str(nclusters))
                    
        data=g.x
        ind = np.argpartition(g.v, -nclusters)[-nclusters:]
        ind =np.sort(ind)
        centroids=data[ind]
        centroids_V=g.v[ind]
            
        g_clusters=G1(N)
        g_clusters.x=np.array(centroids)
        g_clusters.v=np.array(centroids_V)

        selected_cities=np.zeros(nclusters)
        if stepi>0 :
            index1=0
            index2=0
            for k in (ind):
                if k in ind_old:
                    #index2=np.argwhere(ind_old==k)
                    selected_cities[index1]= (selected_cities_n[index2])
                    index2+=1
                index1+=1
        best_selected_cities_N=np.zeros(N)
        index2=0
        for k in range (N):
            if k in ind_old:
                #index=np.argwhere(ind_old==k)
                best_selected_cities_N[k]=selected_cities_n[index2]
                index2+=1
        ind_old=np.copy(ind)
#         best_selected_cities_N=np.array([selected_cities[clx[k]] for k in range (N)])
        if show:
            f = objective_function_simple(N, l, g, best_selected_cities_N,pairwise_distances=None)
            print('f of initializing state = '+str(f))
            
        np.random.seed()
        selected_cities_n, selected_cities_n_convex, loss_values,loss_value_convex = optimize_cluster(N,g_clusters, l, selected_cities, betas=[beta], 
                                                  n_iter=n_iter,mutation_strategy=mutation_strategy, initial_selection_probability=initial_selection_probability, verbose=False)  
        if type(selected_cities_n) == list:
            selected_cities_n = selected_cities_n[-1]
        else:
            print('problem!')

        if show:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            sequence = np.arange(centroids.shape[0])
            axes[1].scatter(centroids[:, 0], centroids[:, 1], c=sequence,label='centroids')
            plotResult(g_clusters,0,selected_cities_n,selected_cities_n_convex,loss_values,loss_value_convex,None,verbose=False)
        
        totalLoss=np.concatenate((totalLoss,loss_values))
        beta=beta+10

#     print('\n Last step, N= '+str(N))
#     selected_cities_n, selected_cities_n_convex, loss_values,loss_value_convex = optimize(g, l, selected_cities=np.array(best_selected_cities_N), beta=beta, n_iter=n_iter,mutation_strategy=mutation_strategy, initial_selection_probability=initial_selection_probability, precompute_pairwise_dist=False, verbose=False)
#     if type(selected_cities_n) == list:
#         selected_cities_n_result = np.array(selected_cities_n[-1])
    
#     totalLoss=np.concatenate((totalLoss,loss_values))
    if show:  
        plotResult(g,0,selected_cities_n,selected_cities_n_convex,loss_values,loss_value_convex,None)
        plt.figure()
        plt.plot(totalLoss)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title('Loss Evolution')  
    totalLoss=totalLoss[::(step_cluster+1)]
    
    #return selected_cities_n_result,selected_cities_n_convex,totalLoss,loss_value_convex
    return selected_cities_n,selected_cities_n_convex,loss_values,loss_value_convex