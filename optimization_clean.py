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


def adding_convex_points(N, l, cities, state):
    selected_cities = state['selected']
    selected_cities_pos = cities.x[selected_cities, :]
    convex_hull = selected_cities_pos[ConvexHull(selected_cities_pos).vertices, :]
    max_distance = np.max(scipy.spatial.distance.pdist(convex_hull, 'sqeuclidean'))
    hull_path = Path(convex_hull)
    is_in_hv = hull_path.contains_points(cities.x)
    # for some known bug in the function, if points is a border is_in_hv can return false
    is_in_hv[selected_cities] = 1
    loss = -np.sum(cities.v[is_in_hv == 1]) + l * N * max_distance * np.pi / 4
    return is_in_hv, loss


def maximum_dist(cities_pos):
    if cities_pos.shape[0] > 2:
        max_distance = np.max(scipy.spatial.distance.pdist(
            cities_pos[ConvexHull(cities_pos).vertices, :], 'sqeuclidean'))
    elif cities_pos.shape[0] == 2:
        max_distance = np.max(scipy.spatial.distance.pdist(cities_pos, 'sqeuclidean'))
    else:
        max_distance = 0.0
    return max_distance


def objective_function(N, l, cities, selected_cities):
    selected_cities_pos = cities.x[selected_cities, :]
    max_distance = maximum_dist(selected_cities_pos)
    return -np.sum(cities.v[selected_cities]) + l * N * max_distance * np.pi / 4


def step(N, cities, state, beta, l, use_kd_tree=True):
    current_loss_value = state['loss_value']
    if np.random.rand() < 0.5:
        if np.random.rand() < 0.95:
            new_center = (state['center'] + np.random.randn(2) * 0.04) % 1
        else:
            new_center = (state['center'] + np.random.randn(2) * 0.2) % 1
        new_radius = state['radius']
    else:
        new_center = state['center']
        new_radius = np.maximum(0.01, state['radius'] + np.random.randn(1) * state['radius'] * 0.1)

    if use_kd_tree:
        selected_cities_k = np.array(state['kdtree'].query_ball_point(new_center, float(new_radius))).astype(np.int32)
    else:
        selected_cities_k = np.where(np.sum((cities.x - new_center) ** 2, axis=1) <= float(new_radius) ** 2)[0]

    if selected_cities_k.shape[0] == 0:
        new_loss_value = np.inf
    else:
        new_loss_value = objective_function(N, l, cities, selected_cities_k)

    a_ik = 1
    if new_loss_value > current_loss_value:
        a_ik = np.exp(-beta * (new_loss_value - current_loss_value))
    accepted = np.random.rand() < a_ik

    if accepted:
        state['selected'] = selected_cities_k
        state['loss_value'] = new_loss_value
        state['center'] = new_center
        state['radius'] = new_radius
    return state


def optimize(cities, l, beta=100, n_iter=20000, verbose=True, use_kd_tree=True):

    if not callable(beta):
        def beta_fn(_1,_2):
            return beta
    else:
        beta_fn = beta

    N = cities.x.shape[0]

    fs = np.zeros(n_iter)

    initial_center = np.random.rand(2)
    initial_radius = np.sqrt(np.random.rand(1) * 0.03 + 0.2)

    # Compute all the cities inside the circle
    if use_kd_tree:
        kd_tree = scipy.spatial.cKDTree(cities.x)
        selected_cities = kd_tree.query_ball_point(initial_center, float(initial_radius))
    else:
        kd_tree = None
        selected_cities = np.where(np.sum((cities.x - initial_center) ** 2, axis=1) <= float(initial_radius) ** 2)[0]

    current_loss_value = objective_function(N, l, cities, selected_cities)
    it = tqdm.notebook.tqdm(range(n_iter)) if verbose else range(n_iter)

    state = {
        'selected': selected_cities,
        'loss_value': current_loss_value,
        'center': initial_center,
        'radius': initial_radius,
        'kdtree': kd_tree
    }

    all_selected_cities = []
    for m in it:
        fs[m] = state['loss_value']
        state = step(N, cities, state, beta_fn(m, n_iter), l, use_kd_tree=use_kd_tree)
        if N <= 20000:
            all_selected_cities.append(state['selected'])

    # Select the lowest error iteration


    if N <= 20000:
        min_idx = np.argmin(fs)
        state['selected'] = all_selected_cities[min_idx]
        state['loss_value'] = fs[min_idx]

        if state['selected'].shape[0] >= 3:
            all_selected_cities_convex, fs_convex = adding_convex_points(N, l, cities, state)
        else:
            all_selected_cities_convex, fs_convex = state['selected'], state['loss_value']

        if fs_convex > fs[-1]:
            print("Convex ", fs_convex, " vs ", state['loss_value'])


    else:
        all_selected_cities.append(state['selected'])

    for i in range(len(all_selected_cities)):
        sel = np.zeros(N, np.int32)
        sel[all_selected_cities[i]] = 1
        all_selected_cities[i] = sel

    # all_selected_cities[-1] = all_selected_cities_convex
    # fs[-1] = fs_convex
    return all_selected_cities, fs
