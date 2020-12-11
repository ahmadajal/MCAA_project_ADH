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
    convex_hull = state['convex_hull']
    hull_path = Path(convex_hull)
    is_in_hv = hull_path.contains_points(cities.x)
    selected_cities = state['selected']
    # for some known bug in the function, if points is a border is_in_hv can return false
    is_in_hv = (is_in_hv | (selected_cities == 1))*1
    max_distance = state['max_dist']  # not changed by adding points in convex hull
    loss = -np.sum(is_in_hv * cities.v) + l * N * max_distance * np.pi / 4
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
    selected_cities_pos = cities.x[selected_cities == 1, :]
    max_distance = maximum_dist(selected_cities_pos)
    return -np.sum(selected_cities * cities.v) + l * N * max_distance * np.pi / 4


def step(N, cities, state, beta, l):
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

    selected_cities_k = (np.sum((cities.x - new_center) ** 2, axis=1) <= new_radius).astype(np.int32)
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


def optimize(cities, l, beta=100, n_iter=20000, verbose=True):

    if not callable(beta):
        def beta_fn(_):
            return beta
    else:
        beta_fn = beta

    N = cities.x.shape[0]

    fs = np.zeros(n_iter)
    all_selected_cities = []

    initial_center = np.random.rand(2)
    initial_radius = np.random.rand(1) * 0.03 + 0.2

    # Compute all the cities inside the circle
    selected_cities = (np.sum((cities.x - initial_center) ** 2, axis=1) <= initial_radius).astype(np.int32)

    current_loss_value = objective_function(N, l, cities, selected_cities)
    it = tqdm.notebook.tqdm(range(n_iter)) if verbose else range(n_iter)

    state = {
        'selected': selected_cities,
        'loss_value': current_loss_value,
        'center': initial_center,
        'radius': initial_radius
    }

    for m in it:
        fs[m] = state['loss_value']
        state = step(N, cities, state, beta_fn(m), l)
        all_selected_cities.append(state['selected'])

    selected_cities_pos = cities.x[state['selected'] == 1, :]
    if np.sum(state['selected']) >= 3:
        state['convex_hull'] = selected_cities_pos[ConvexHull(selected_cities_pos).vertices, :]
    state['max_dist'] = maximum_dist(selected_cities_pos)

    if np.sum(state['selected']) >= 3:
        all_selected_cities_convex, fs_convex = adding_convex_points(N, l, cities, state)
    else:
        all_selected_cities_convex, fs_convex = state['selected'], state['loss_value']

    if fs_convex > fs[-1]:
        print("Convex ", fs_convex, " vs ", fs[-1])

    all_selected_cities[-1] = all_selected_cities_convex
    fs[-1] = fs_convex
    return all_selected_cities, fs
