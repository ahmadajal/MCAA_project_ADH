import numpy as np
import scipy
import tqdm
import tqdm.notebook

from util import *
import neighbors
import baseline

"""This file contains the implementation of our best optimization strategy based on a continuous Markov chain
   modifying the parameters of a circle"""


def get_cities_in_circle(cities, kd_tree, center, radius):
    """Return all the cities inside a given circle. Uses a KD-tree if provided"""

    if kd_tree is not None:
        selected = np.array(kd_tree.query_ball_point(center, float(radius))).astype(np.int32)
    else:
        selected = np.where(np.sum((cities.x - center) ** 2, axis=1) <= float(radius) ** 2)[0]

    selected_ret = np.zeros(cities.x.shape[0], np.int32)
    selected_ret[selected] = 1
    return selected_ret


def step(cities, state, beta, l, rng):
    current_loss_value = state['loss_value']
    if rng.rand() < 0.5:
        if rng.rand() < 0.95:
            new_center = (state['center'] + rng.randn(2) * 0.04) % 1
        else:
            new_center = (state['center'] + rng.randn(2) * 0.2) % 1
        new_radius = state['radius']
    else:
        new_center = state['center']
        new_radius = np.maximum(0.01, state['radius'] + rng.randn(1) * state['radius'] * 0.1)

    selected_cities_k = get_cities_in_circle(cities, state['kdtree'], new_center, new_radius)
    new_loss_value = objective_function(l, cities, selected_cities_k)
    accepted = rng.rand() < np.minimum(1, np.exp(-beta * (new_loss_value - current_loss_value)))

    if accepted:
        state['selected'] = selected_cities_k
        state['loss_value'] = new_loss_value
        state['center'] = new_center
        state['radius'] = new_radius
    return state


def optimize(cities, l, beta, n_iter, verbose=True, rng=None):
    if rng is None:
        rng = np.random

    use_kd_tree = True
    beta_fn = create_beta_fun(beta)
    initial_center = rng.rand(2)
    initial_radius = np.sqrt(rng.rand(1) * 0.2 + 0.1)

    # max_index = np.argmax(cities.v)
    # initial_center = cities.x[max_index, :]

    kd_tree = scipy.spatial.cKDTree(cities.x) if use_kd_tree else None
    selected_cities = get_cities_in_circle(cities, kd_tree, initial_center, initial_radius)
    current_loss_value = objective_function(l, cities, selected_cities)
    state = {'selected': selected_cities, 'loss_value': current_loss_value,
             'center': initial_center, 'radius': initial_radius, 'kdtree': kd_tree}

    # Ratio of iterations used for smooth version, the rest of iterations will use a flipping strategy
    ratio = 0.85
    if ratio < 1:
        state['delaunay'] = scipy.spatial.Delaunay(cities.x)

    best_loss = np.inf
    number_of_selected_cities = []
    loss_values = np.zeros(n_iter)
    it = tqdm.notebook.tqdm(range(n_iter)) if verbose else range(n_iter)
    for m in it:
        loss_values[m] = state['loss_value']
        number_of_selected_cities.append(len(state['selected']))
        if m == np.round(ratio * n_iter):
            state['selected'] = best_selection
            state['loss_value'] = best_loss

        if m < ratio * n_iter:
            state = step(cities, state, beta_fn(m, n_iter), l, rng=rng)
        else:
            state = neighbors.step(cities, state, 15, l, rng=rng)
        if state['loss_value'] < best_loss:
            best_loss = state['loss_value']
            best_selection = state['selected']

    state['loss_value'] = best_loss
    state['selected'] = best_selection
    if np.sum(state['selected']) >= 3:
        best_selection, best_loss = add_points_in_convex_hull(l, cities, best_selection)

    loss_values[-1] = best_loss
    number_of_selected_cities[-1] = len(best_selection)
    return best_selection, loss_values, number_of_selected_cities
