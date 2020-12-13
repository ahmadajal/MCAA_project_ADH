import matplotlib.pyplot as plt
from matplotlib.path import Path

import numpy as np

import scipy
import scipy as sp
import scipy.stats as st
from scipy.spatial import ConvexHull

import tqdm
import tqdm.notebook
from sklearn.cluster import KMeans
import copy

class DatasetGenerator(object):
    def __init__(self, N=100):
        self.N = N
        self.x = None
        self.v = None
        self.refresh()

    def refresh(self):
        raise Exception("undefined")


class G1(DatasetGenerator):
    def refresh(self):
        self.x = st.uniform().rvs((self.N, 2))
        self.v = st.uniform().rvs((self.N,))
    def __str__(self):
        return "G1"


class G2(DatasetGenerator):
    def refresh(self):
        self.x = st.uniform().rvs((self.N, 2))
        self.v = np.exp(st.norm(-0.85, 1.3).rvs((self.N,)))
    def __str__(self):
        return "G2"

def find_most_dense_cluster(cities, n_cluster=10):
    # Finds the cluster of cities with maximum density
    kmeans = KMeans(n_clusters=n_cluster, max_iter=500)
    kmeans.fit(cities.x)
    density = [sum(cities.v[np.where(kmeans.labels_==k)[0]])/len(np.where(kmeans.labels_==k)[0]) for k in range(n_cluster)]
    max_density_cities_idx = np.where(kmeans.labels_==np.argmax(density))[0]
    max_density_cluster_center = kmeans.cluster_centers_[np.argmax(density)]
    max_radius = np.max(np.sqrt(np.sum((cities.x[max_density_cities_idx] - max_density_cluster_center)**2, axis=1)))
    return max_density_cluster_center, max_radius, max_density_cities_idx


# an optimal beta as said in the course, could be interesting to use it
def compute_beta(N0, N1, f0, f1, epsilon):
    return np.log(N1 / (epsilon * N0)) / (f1 - f0)


#add points that are in the convex hull
def adding_convex_points(N, l, cities, state):
    convex_hull = state['convex_hull']
    hull_path = Path(convex_hull)
    is_in_hv = hull_path.contains_points(cities.x)
    selected_cities = state['selected']
    # for some known bug in the function, if points is a border is_in_hv can return false
    is_in_hv = (is_in_hv | (selected_cities == 1))*1
    max_distance = state['max_dist']  # not changed by adding points in convex hull
    loss = -np.sum(is_in_hv * cities.v) + l * N * max_distance * np.pi / 4
    return is_in_hv, loss

def adding_convex_points_loss(N, l, cities,selected_cities,convex_hull,max_distance):
    # Delio: Recomputing all these quantities seems to fix the issues
    # selected_cities_pos = cities.x[selected_cities == 1, :]
    # convex_hull = selected_cities_pos[ConvexHull(selected_cities_pos).vertices, :]
    # max_distance = np.max(scipy.spatial.distance.pdist(convex_hull, 'sqeuclidean'))


    hull_path = Path(convex_hull)
    is_in_hv=hull_path.contains_points(cities.x)
    is_in_hv=(is_in_hv | (selected_cities == 1))*1 # for some known bug in the function, if points is a border is_in_hv can return false
    loss = -np.sum(is_in_hv * cities.v) + l * N * max_distance * np.pi / 4
    return loss

#indexing code from https://stackoverflow.com/a/36867493/2351867
def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))//2

def condensed_to_square(k, n):
    i = int(np.ceil((1/2.) * (- (-8*k + 4 * n**2 - 4*n - 7)**0.5 + 2*n - 1) - 1))
    j = int(n - elem_in_i_rows(i + 1, n) + k)
    return i, j

def square_to_condensed(i, j, n):
    if i==j:
        print('problem !')
        print (i)
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return n*j - j*(j+1)//2 + i - 1 - j

def square_to_condensed_list(I, J, n):
    l=[]
    for i in I:
        for j in J: 
            assert i != j, "no diagonal elements in condensed matrix"
            i2,j2=i,j
            if i < j:
                i2, j2 = j, i
            l.append(n*j2 - j2*(j2+1)//2 + i2 - 1 - j2)
    return l


def maximum_dist(cities_pos):
    if cities_pos.shape[0] > 2:
        max_distance = np.max(scipy.spatial.distance.pdist(
            cities_pos[ConvexHull(cities_pos).vertices, :], 'sqeuclidean'))
    elif cities_pos.shape[0] == 2:
        max_distance = np.max(scipy.spatial.distance.pdist(cities_pos, 'sqeuclidean'))
    else:
        max_distance = 0.0

    return max_distance

# Original efficient objective evaluation, currently unused
def objective_function_simple(N, l, cities, selected_cities, pairwise_distances):
    selected_cities_pos = cities.x[selected_cities == 1, :]
    if pairwise_distances is not None:
        max_distance = np.max(np.outer(selected_cities, selected_cities) * pairwise_distances)
    else:
        # Compute the maximum distance by computing the distances over the convex hull vertices
        max_distance = maximum_dist(selected_cities_pos)

    return -np.sum(selected_cities * cities.v) + l * N * max_distance * np.pi / 4


# This function computes the objective value given some current `state`
# and new set selected_cities, changed at `change_idx`
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


def objective_function(N, l, cities, selected_cities, pairwise_distances):
    return objective_function_(N, l, cities, None, selected_cities, None, pairwise_distances)


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
            new_loss_value = adding_convex_points_loss(N, l, cities, selected_cities_k, new_convex_hull, new_max_dist)
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


def optimize(cities, l, beta=100, n_iter=20000, mutation_strategy=0, initial_selection_probability=0.5, precompute_pairwise_dist=False, verbose=True, selected_cities=None):
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
        all_selected_cities_convex, fs_convex = adding_convex_points(N, l, cities, state)
    else:
        all_selected_cities_convex, fs_convex = state['selected'], state['loss_value']
    return all_selected_cities, all_selected_cities_convex, fs, fs_convex


def optimize_with_initialize(cities, l, selected_cities_init, beta=100, n_iter=20000, mutation_strategy=0, initial_selection_probability=0.5, precompute_pairwise_dist=False, verbose=True):
    """mutation_strategy = 0: Original mutation proposed by Heloise
       mutation_strategy = 1: Simple strategy which just randomly tries to flip cities
       initial_selection_probability: Probablity at which a city initially is selected (0.5: every city can be selected with 50% chance)

       precompute_pairwise_dist: Enabling this gives slightly better performance, but has quadratic memory complexity
       """

    return optimize(cities, l, beta=beta, n_iter=n_iter,
                    mutation_strategy=mutation_strategy, initial_selection_probability=initial_selection_probability,
                    precompute_pairwise_dist=precompute_pairwise_dist,
                    verbose=verbose, selected_cities=selected_cities_init)

    N = cities.x.shape[0]
    selected_cities=selected_cities_init
#     print("done")
    if precompute_pairwise_dist:
        pairwise_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(cities.x, 'sqeuclidean'))
    else:
        pairwise_distances = None

    fs = np.zeros(n_iter)
    all_selected_cities = []
    current_loss_value, max_dist, max_idx, convex_hull = objective_function_(N, l, cities, None, selected_cities, None, pairwise_distances)
    it = tqdm.notebook.tqdm(range(n_iter)) if verbose else range(n_iter)

    state = {
        'selected': selected_cities,
        'loss_value': current_loss_value,
        'max_dist': max_dist,
        'max_idx': max_idx,
        'convex_hull': convex_hull,
        # 'max_points': max_points,
    }
    for m in it:
        fs[m] = state['loss_value']
        state = step(N, cities, state, beta, l, pairwise_distances,
            mutation_strategy=mutation_strategy)
        all_selected_cities.append(state['selected'])
    all_selected_cities_convex,fs_convex=adding_convex_points(N, l, cities,state)
    return all_selected_cities, all_selected_cities_convex,fs,fs_convex


def optimize_with_initialize_betas(cities, l, selected_cities_init, betas=[20,100], n_iter=20000, mutation_strategy=0, initial_selection_probability=0.5, precompute_pairwise_dist=False, verbose=True):
    """mutation_strategy = 0: Original mutation proposed by Heloise
       mutation_strategy = 1: Simple strategy which just randomly tries to flip cities
       initial_selection_probability: Probablity at which a city initially is selected (0.5: every city can be selected with 50% chance)

       precompute_pairwise_dist: Enabling this gives slightly better performance, but has quadratic memory complexity
       """

    N = cities.x.shape[0]
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


def number_min_city(cities,l):
    N = cities.x.shape[0]
    maxV=np.amax(cities.v)
    convex_hull = ConvexHull(cities.x)
    convex_hull_dists = scipy.spatial.distance.pdist(cities.x[convex_hull.vertices, :], 'sqeuclidean')
    min_dist_idx = np.argmin(convex_hull_dists)
    min_distance = convex_hull_dists[min_dist_idx]
    k_min=l*N*np.pi*min_distance*min_distance/(4*maxV)
    return k_min



#------------------Old---------------------------

def objectiveFunction_old(N, l, citiesV, selectedCities, pairwise_distances):
    # Compute maximum area
    max_area = np.pi / 4 * np.max(np.outer(selectedCities, selectedCities) * pairwise_distances)

    # Compute final loss
    f = np.sum(selectedCities * citiesV) - l * N * max_area
    return -f

def acceptancePb(selectedCities_i,selectedCities_j,beta,N,l,citiesV, pairwise_distances):
    fi = objectiveFunction_old(N, l, citiesV, selectedCities_i, pairwise_distances)
    fj = objectiveFunction_old(N, l, citiesV, selectedCities_j, pairwise_distances)
    result = np.exp(-beta * (fj - fi))
    return min(1, result)

def step_old(N, citiesX, citiesV, selectedCities_i, beta, l, pairwise_distances):
    k = np.random.randint(0, N);
    if np.random.rand() < 0.5: # Remove a city
        if selectedCities_i[k] == 0: # same state then before
            return selectedCities_i # do nothing, it is accepted
        else:
            selectedCities_k = np.copy(selectedCities_i)
            selectedCities_k[k] = 0 # city removed from set
            a_ik = acceptancePb(selectedCities_i, selectedCities_k, beta, N, l, citiesV, pairwise_distances)
            if np.random.rand() < a_ik:
                return selectedCities_k #accepted!
            else:
                return selectedCities_i #refused
    else: # Add a city
        if selectedCities_i[k] == 1: # do nothing, city already in set
            return selectedCities_i
        else:
            selectedCities_k = np.copy(selectedCities_i)
            selectedCities_k[k] = 1 # add city to set
            #could of course be computed in a smarter way
            a_ik = acceptancePb(selectedCities_i, selectedCities_k, beta, N, l, citiesV, pairwise_distances)
            if np.random.rand() < a_ik:
                return selectedCities_k #city added!
            else:
                return selectedCities_i #refused

def optimize_old(cities, l, beta=100, n_iter=20000, initial_selection_probability=0.5, precompute_pairwise_dist=False, verbose=True):
    N = cities.x.shape[0]
    selected_cities = (np.random.rand(N) <= initial_selection_probability).astype(np.int32)

    pairwise_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(cities.x, 'sqeuclidean'))


    fs = np.zeros(n_iter) #keep record of objective function (in fact, minus objective function)
    all_selected_cities = []

    it = tqdm.notebook.tqdm(range(n_iter)) if verbose else range(n_iter)

    for m in it:
        fs[m] = objectiveFunction_old(N, l, cities.v, selected_cities, pairwise_distances)
        selected_cities = step_old(N, cities.x, cities.v, selected_cities, beta, l, pairwise_distances)
        all_selected_cities.append(selected_cities)

    return all_selected_cities,fs


def optimize_combine(g, l, selected_cities_init, betas=[20,100], n_iter=20000, precompute_pairwise_dist=False, verbose=True):
    selected_cities_n, selected_cities_n_convex, loss_values1,loss_value_convex = optimize_with_initialize_betas(g, l,selected_cities_init, betas=betas,  n_iter=n_iter,mutation_strategy=3, precompute_pairwise_dist=False, verbose=False)
    loss_values1[-1]=(loss_value_convex)  
    if type(selected_cities_n) == list:
        selected_cities_n = selected_cities_n[-1]
    selected_cities_n, selected_cities_n_convex, loss_values,loss_value_convex = optimize_with_initialize_betas(g, l, selected_cities_n_convex, betas=[betas[-1]],n_iter=n_iter,mutation_strategy=2,precompute_pairwise_dist=False, verbose=False)
    if type(selected_cities_n) == list:
        selected_cities_n = selected_cities_n[-1]
    loss_values[-1]=(loss_value_convex) 
    total_loss=np.concatenate((loss_values1,loss_values))
    total_loss=total_loss[::2]
    return selected_cities_n,selected_cities_n_convex,total_loss,loss_value_convex


# Old code
# Currently unused: We now use the objective function evaluation from the previous iteration to
# improve performance
# def acceptance_pb(selected_cities_i, selected_cities_j, beta, N, l, cities, pairwise_distances):
#     fi = objective_function(N, l, cities, selected_cities_i, pairwise_distances)
#     fj = objective_function(N, l, cities, selected_cities_j, pairwise_distances)
#     result = np.exp(-beta * (fj - fi))
#     return min(1, result)

# def pbij(selected_cities_i, selected_cities_j, beta, N, l, citiesV):
#     # not used in the algorithm, could be useful to plot and compare statistics
#     if selected_cities_i == selected_cities_j:
#         s = 0
#         for k in range(N):
#             selected_cities_k = np.copy(selected_cities_i)
#             selected_cities_k[k] = 1-selected_cities_i[k]
#             a_ik = acceptance_pb(selected_cities_i, selected_cities_k, beta, n, l, citiesV)
#             phi_ik = (1/2)*(1/N)
#             s += phi_ik*a_ik
#         return 1-s
#     else:
#         a_ij = acceptance_pb(selected_cities_i, selected_cities_j, beta, n, l, citiesV)
#         phi_ij = (1/2)*(1/N)  # pb of chosing a city*pb choosing if 0 or 1
#         return a_ij*phi_ij
