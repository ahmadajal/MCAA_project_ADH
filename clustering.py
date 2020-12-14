import matplotlib.pyplot as plt
from matplotlib.path import Path

import numpy as np

import scipy
import scipy as sp
import scipy.stats as st
from scipy.spatial import ConvexHull

import tqdm
import tqdm.notebook

#from optimization import *
#from optimal_solution import *
import baseline as base

import time
import scipy
from scipy import spatial
import itertools

from scipy.cluster.vq import kmeans,vq,whiten

import util


def bruteforce_sol_N(N,N2,l, generator,pairwise_distances):
    # very inefficient solution: O(2^N)
    min_f = np.inf
    all_combination_cities = list(map(list, itertools.product([0, 1], repeat=N)))
    if pairwise_distances is None:
        pairwise_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(generator.x, 'sqeuclidean'))
    for selected_cities in all_combination_cities[1:]:
        f = objective_function(N2, l, generator, selected_cities)
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
        all_selected_cities_convex,fs_convex=util.add_points_in_convex_hull(l, cities,state['selected'])
    return all_selected_cities, all_selected_cities_convex,fs,fs_convex


def optimize(g, l, beta=20, n_iter=10000, beta_last=40, mutation_strategy=1 , verbose=True, show=False):
    #set number of steps
    N=g.x.shape[0]
    step_cluster=np.int(np.floor(np.log10(N)))
    true_step_cluster=min(3,np.int(np.floor(np.log10(N)))-1)
    print('number of steps= '+str(step_cluster))
    betas=[beta]
    totalLoss=[]
    mutation_strategy=1
    betas_init=beta
    

    for stepi in range (0,true_step_cluster):
        #betas=[betas[0]+10]
        betas=[betas_init+stepi*(beta_last-betas_init)/true_step_cluster]
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
        g_clusters=util.G1(N)
        g_clusters.x=np.array(centroids)
        g_clusters.v=np.array(centroids_V)
             
        if stepi>0 :
            #selected_cities=[np.sum((clx==k)*(is_in_selected_cluster))==(sum(clx==k)/2) for k in range (nclusters)]
            selected_cities=np.array(selected_cities)
            for k in range (nclusters):
                in_cluster=clx==k
                is_selected_city=is_in_selected_cluster
                good_cities=in_cluster*is_in_selected_cluster
                if np.sum(good_cities)==(sum(in_cluster)):
                    selected_cities[k]=1
                    
        best_selected_cities_N=np.array([selected_cities[clx[k]] for k in range (N)])
        if show:
            f = objective_function(N, l, g, best_selected_cities_N)
            print('f of initializing state = '+str(f))
            
        np.random.seed()
        selected_cities_n, selected_cities_n_convex, loss_values,loss_value_convex = optimize_cluster(N,g_clusters, l, selected_cities, betas=betas, 
                                                  n_iter=n_iter,mutation_strategy=mutation_strategy, verbose=False)  
        if type(selected_cities_n) == list:
            selected_cities_n = selected_cities_n[-1]
        else:
            print('problem!')

        if show:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            axes[0].scatter(g.x[:,0],g.x[:,1],c=clx ,label='cities')
            sequence = np.arange(centroids.shape[0])
            axes[1].scatter(centroids[:, 0], centroids[:, 1], c=sequence,label='centroids')
            plotResult(g_clusters,0,selected_cities_n,selected_cities_n_convex,loss_values,loss_value_convex,None,verbose=False)
        
        totalLoss=np.concatenate((totalLoss,loss_values))

    print('\n Last step, N= '+str(N))
    is_in_selected_cluster=[selected_cities_n[clx[k]]==1 for k in range (N)]
    selected_cities_n, selected_cities_n_convex, loss_values,loss_value_convex = optimize_baseline(g, l, beta=beta_last, n_iter=n_iter ,selected_cities=np.array(is_in_selected_cluster), verbose=False)
    if type(selected_cities_n) == list:
        selected_cities_n_result = np.array(selected_cities_n[-1])
    
    totalLoss=np.concatenate((totalLoss,loss_values))
    if show:  
        plotResult(g,0,selected_cities_n_convex,selected_cities_n_convex,loss_values,loss_value_convex,None)
        plt.figure()
        plt.plot(totalLoss)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title('Loss Evolution')  
    totalLoss=totalLoss[::(true_step_cluster+1)]
    
    loss_values[-1]=loss_value_convex
    number_of_selected_cities=np.zeros(loss_values.shape[0]) +np.sum(selected_cities_n_convex)
    return selected_cities_n_convex,loss_values,number_of_selected_cities    

    #return selected_cities_n_result,selected_cities_n_convex,loss_values,loss_value_convex
    #return selected_cities_n_result,selected_cities_n_convex,totalLoss,loss_value_convex


def optimize_simple(g, l, beta=20, beta_last=40, n_iter=10000, mutation_strategy=1, verbose=True, show=False):
    #set number of steps
    N=g.x.shape[0]
    step_cluster=np.int(np.floor(np.log10(N)))
    print('number of steps= '+str(step_cluster))
    beta=beta
    totalLoss=[]
    ind_old=[]
    best_selected_cities_N=np.zeros(N)
    mutation_strategy=1
    betas_init=beta
    #beta_fn = util.create_beta_fun(beta)

    for stepi in range (0,step_cluster):
        nclusters=N//(10**(step_cluster-1-stepi))
        print('step = '+str(stepi)+', N clusters= '+str(nclusters))
        beta=betas_init+stepi*(beta_last-betas_init)/step_cluster       
        data=g.x
        ind = np.argpartition(g.v, -nclusters)[-nclusters:]
        ind =np.sort(ind)
        centroids=data[ind]
        centroids_V=g.v[ind]
            
        g_clusters=util.G1(N)
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
            f = objective_function(N, l, g, best_selected_cities_N)
            print('f of initializing state = '+str(f))
            
        np.random.seed()
        selected_cities_n, selected_cities_n_convex, loss_values,loss_value_convex = optimize_cluster(N,g_clusters, l, selected_cities, betas=[beta], 
                                                  n_iter=n_iter,mutation_strategy=mutation_strategy, verbose=False)  
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
    #return selected_cities_n,selected_cities_n_convex,loss_values,loss_value_convex
    
    loss_values[-1]=loss_value_convex
    number_of_selected_cities=np.zeros(loss_values.shape[0]) +np.sum(selected_cities_n_convex)
    return selected_cities_n_convex,loss_values,number_of_selected_cities



def objective_function(N,l, cities, selected_cities):
    if len(selected_cities) == 0:
        return np.inf
    selected_cities_pos = cities.x[selected_cities == 1, :]
    max_distance = util.maximum_dist(selected_cities_pos)
    return -np.sum(cities.v[selected_cities == 1]) + l * N * max_distance * np.pi / 4


def objective_function_(N, l, cities, state, selected_cities, change_idx, pairwise_distances):
    if pairwise_distances is not None:
        table= np.outer(selected_cities, selected_cities) * pairwise_distances
        max_distance = np.max(table)
        max_indices = np.where(table == max_distance)
        convex_hull=None
    else:
        selected_indices = np.where(selected_cities == 1)[0]
        selected_cities_pos = cities.x[selected_indices, :]

        if state is not None:
            max_distance = state['max_dist']
            max_indices = state['max_idx']
            convex_hull = state['convex_hull']

        # special cases for no or 2 cities being selected
        if selected_cities_pos.shape[0] <= 1:
            max_distance = 0
            max_indices = (0, 0)
            convex_hull = selected_cities_pos
        elif selected_cities_pos.shape[0] == 2:
            max_distance = np.sum((selected_cities_pos[0] - selected_cities_pos[1]) ** 2)
            max_indices = (0, 1)
            convex_hull = selected_cities_pos
        elif (change_idx is not None) and (selected_cities[change_idx] == 1):
            # Adding a vertex: Just need to compute its distance to all selected vertices
            # to see if te maximum distance became larger
            distances = np.sum((cities.x[change_idx] - selected_cities_pos) ** 2, axis=1)
            max_dist_idx = np.argmax(distances)
            max_distance_new = distances[max_dist_idx]
            if max_distance_new > state['max_dist']:
                max_distance = max_distance_new
                max_indices = [change_idx, selected_indices[max_dist_idx]]
            convex_hull = ConvexHull(selected_cities_pos) #convex hull can change if a city is added
            convex_hull = selected_cities_pos[convex_hull.vertices, :]
        elif ((change_idx is not None) and (selected_cities[change_idx] == 0) and change_idx in state['max_idx']) or change_idx is None:
            # Recompute distance either if we remove on of the vertices used in distance computation or
            # if no change idx was defined (e.g. at the start of the algorithm)
            # Compute the maximum distance by computing the distances over the convex hull vertices
            convex_hull = ConvexHull(selected_cities_pos)
            convex_hull_dists = scipy.spatial.distance.pdist(selected_cities_pos[convex_hull.vertices, :], 'sqeuclidean')
            max_dist_idx = np.argmax(convex_hull_dists)
            max_distance = convex_hull_dists[max_dist_idx]
            convex_hull_indices = condensed_to_square(max_dist_idx, convex_hull.vertices.shape[0])
            max_indices = [convex_hull.vertices[convex_hull_indices[0]], convex_hull.vertices[convex_hull_indices[1]]]
            max_indices = [selected_indices[max_indices[0]], selected_indices[max_indices[1]]]
            convex_hull = selected_cities_pos[convex_hull.vertices, :]

    loss = -np.sum(selected_cities * cities.v) + l * N * max_distance * np.pi / 4
    return loss, max_distance, max_indices, convex_hull



def step(N, cities, state, beta, l, pairwise_distances, mutation_strategy=0):
    n_selected = np.sum(state['selected'])

    if mutation_strategy == 5:
        current_loss_value = state['loss_value']

        # take a step for both position and radius

        if np.random.rand() < 0.5:
            if np.random.rand() < 0.95:
                new_center = (state['center'] + np.random.randn(2) * 0.04) % 1
            else:
                new_center = (state['center'] + np.random.randn(2) * 0.2) % 1
            new_radius = state['radius']
        else:
            new_center = state['center']
            new_radius = np.maximum(0.01, state['radius'] + np.random.randn(1) * state['radius'] * 0.1)

        # Compute all the cities inside the circle
        selected_cities_k = (np.sum((cities.x - new_center) ** 2, axis=1) <= new_radius).astype(np.int32)

        new_loss_value = objective_function_simple(N, l, cities, selected_cities_k, pairwise_distances)
        if selected_cities_k.shape[0] == 0:
            new_loss_value = np.inf

        a_ik = 1
        if new_loss_value > current_loss_value:
            a_ik = np.exp(-beta * (new_loss_value - current_loss_value))
        accepted = np.random.rand() < a_ik

        new_state = copy.deepcopy(state)

        if accepted:
            new_state['selected'] = selected_cities_k
            new_state['loss_value'] = new_loss_value
            new_state['center'] = new_center
            new_state['radius'] = new_radius

        return new_state

    if mutation_strategy == 4 and n_selected > 0 and np.random.rand() < 0.5:
        # Randomly sample one of the selected cities
        k = np.random.randint(0, n_selected)
        sampled = np.where(state['selected'] == 1)[0][k]

        tri = state['delaunay']
        indptr, indices = tri.vertex_neighbor_vertices
        # nn = np.append(indices[indptr[sampled]:indptr[sampled + 1]], [sampled]).ravel()
        nn = indices[indptr[sampled]:indptr[sampled + 1]]
        k = nn[np.random.randint(0, nn.shape[0])]
    else:
        k = np.random.randint(0, cities.x.shape[0])
    remove_city = np.random.rand() < 0.5

    selected_cities_i = state['selected']
    if mutation_strategy == 2:  # flip
        remove_city = (1 - selected_cities_i[k]) == 0
    current_loss_value = state['loss_value']
    if (mutation_strategy == 0) and ((remove_city and selected_cities_i[k] == 0) or (not remove_city and selected_cities_i[k] == 1)):
        return state  # do nothing
    else:
        selected_cities_k = np.copy(selected_cities_i)
        selected_cities_k[k] = 1 - selected_cities_k[k]
        new_loss_value, new_max_dist, new_max_idx, new_convex_hull = objective_function_(
            N, l, cities, state, selected_cities_k, k, pairwise_distances)
        if mutation_strategy == 3 and (np.where(selected_cities_k == 1)[0]).shape[0] > 3:
            _,new_loss_value = util.add_points_in_convex_hull(l, cities, selected_cities_k)
        a_ik = 1
        if new_loss_value > current_loss_value:  # less computation
            a_ik = np.exp(-beta * (new_loss_value - current_loss_value))
        accepted = np.random.rand() < a_ik
        new_state = {
            'selected': selected_cities_k if accepted else state['selected'],
            'loss_value': new_loss_value if accepted else state['loss_value'],
            'max_dist': new_max_dist if accepted else state['max_dist'],
            'max_idx': new_max_idx if accepted else state['max_idx'],
            'convex_hull': new_convex_hull if accepted else state['convex_hull'],
            'delaunay': state['delaunay']
        }
        return new_state


def optimize_baseline(cities, l, beta=100, n_iter=20000, mutation_strategy=0, initial_selection_probability=0.5, precompute_pairwise_dist=False, verbose=True, selected_cities=None):
    """mutation_strategy = 0: Original mutation proposed by Heloise
       mutation_strategy = 1: Simple strategy which just randomly tries to flip cities
       initial_selection_probability: Probablity at which a city initially is selected (0.5: every city can be selected with 50% chance)

       precompute_pairwise_dist: Enabling this gives slightly better performance, but has quadratic memory complexity
	selected_cities: Allows to pass in an initial selection of cities
       """
    # Allow beta to be a function depending on the iteration count i
    if not callable(beta):
        def beta_fn(_1,_2): return beta
    else:
        beta_fn = beta

    N = cities.x.shape[0]
    if selected_cities is None:
        selected_cities = (np.random.rand(N) <= initial_selection_probability).astype(np.int32)

    if precompute_pairwise_dist:
        pairwise_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(cities.x, 'sqeuclidean'))
    else:
        pairwise_distances = None

    fs = np.zeros(n_iter)
    all_selected_cities = []

    if mutation_strategy == 5:
        initial_center = np.random.rand(2)
        initial_radius = np.random.rand(1) * 0.03 + 0.2
        # Compute all the cities inside the circle
        selected_cities = (np.sum((cities.x - initial_center) ** 2, axis=1) <= initial_radius).astype(np.int32)


    current_loss_value, max_dist, max_idx, convex_hull = objective_function_(
        N, l, cities, None, selected_cities, None, pairwise_distances)
    it = tqdm.notebook.tqdm(range(n_iter)) if verbose else range(n_iter)

    state = {
        'selected': selected_cities,
        'loss_value': current_loss_value,
        'max_dist': max_dist,
        'max_idx': max_idx,
        'convex_hull': convex_hull,
        # 'max_points': max_points,
    }
    state['delaunay'] = scipy.spatial.Delaunay(cities.x) if  mutation_strategy == 4 else None

    if mutation_strategy == 5:
        state['center'] = initial_center
        state['radius'] = initial_radius

    for m in it:
        fs[m] = state['loss_value']
        state = step(N, cities, state, beta_fn(m, n_iter), l, pairwise_distances,
                     mutation_strategy=mutation_strategy)

        # Delio: Recomputing the convex_hull for strategy 3 seems to fix some issues?
        # if mutation_strategy == 3:
        #     selected_cities_pos = cities.x[state['selected'] == 1, :]
        #     if selected_cities_pos.shape[0] >= 3:
        #         state['convex_hull'] = selected_cities_pos[ConvexHull(selected_cities_pos).vertices, :]

        all_selected_cities.append(state['selected'])

    if mutation_strategy == 5:
        selected_cities_pos = cities.x[state['selected'] == 1, :]
        if np.sum(state['selected']) >= 3:
            state['convex_hull'] = selected_cities_pos[ConvexHull(selected_cities_pos).vertices, :]
        state['max_dist'] = maximum_dist(selected_cities_pos)

    if np.sum(state['selected']) >= 3:
        all_selected_cities_convex, fs_convex = util.add_points_in_convex_hull(l, cities, state['selected'])
    else:
        all_selected_cities_convex, fs_convex = state['selected'], state['loss_value']
    return all_selected_cities, all_selected_cities_convex, fs, fs_convex
    
    
def condensed_to_square(k, n):
    i = int(np.ceil((1/2.) * (- (-8*k + 4 * n**2 - 4*n - 7)**0.5 + 2*n - 1) - 1))
    j = int(n - elem_in_i_rows(i + 1, n) + k)
    return i, j

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))//2


