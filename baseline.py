import numpy as np
import scipy
import tqdm
import tqdm.notebook

from util import *

"""This file contains the implementation of our baseline Metropolis implementation based on
   just randomly adding and removing individual cities"""

def step(cities, state, beta, l):
    current_loss_value = state['loss_value']
    k = np.random.randint(cities.x.shape[0])
    selected_cities_k = np.array(state['selected'])
    selected_cities_k[k] = 1 - selected_cities_k[k]
    new_loss_value = objective_function(l, cities, selected_cities_k)
    accepted = np.random.rand() < np.minimum(1, np.exp(-beta * (new_loss_value - current_loss_value)))
    if accepted:
        state['selected'] = selected_cities_k
        state['loss_value'] = new_loss_value
    return state


def optimize(cities, l, beta, n_iter, verbose=True):
    beta_fn = create_beta_fun(beta)
    selected_cities = np.zeros(cities.x.shape[0], np.int32)
    current_loss_value = objective_function(l, cities, selected_cities)
    state = {'selected': selected_cities, 'loss_value': current_loss_value}
    best_loss = np.inf
    number_of_selected_cities = []
    loss_values = np.zeros(n_iter)
    it = tqdm.notebook.tqdm(range(n_iter)) if verbose else range(n_iter)
    for m in it:
        loss_values[m] = state['loss_value']
        number_of_selected_cities.append(len(state['selected']))
        state = step(cities, state, beta_fn(m, n_iter), l)
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
