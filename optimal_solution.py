import numpy as np
import matplotlib.pyplot as plt

import tqdm 
import tqdm.notebook

from optimization import *

import time

def bruteforce_sol(N, l, generator):
    # very inefficient solution: O(2^N)
    min_f = np.inf
    all_combination_cities = list(map(list, itertools.product([0, 1], repeat=N)))
    pairwise_distances = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(generator.x, 'sqeuclidean'))
    for selected_cities in all_combination_cities[1:]:
        f = objective_function_simple(N, l, generator, selected_cities, 
                                      pairwise_distances=pairwise_distances)
        if f < min_f:
            solution = selected_cities
            min_f = f
    return solution, min_f


def opt_solution(N, l, generator, verbose=False):
    # better solution: O(N*3)
    min_f = np.inf
    pairwise_distances = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(generator.x, 'sqeuclidean'))
    it = tqdm.notebook.tqdm(range(N-1)) if verbose else range(N-1)
    for first_city in it:
        for second_city in range(first_city, N):
            selected_cities = np.zeros(N, dtype=int)
            max_dist = scipy.spatial.distance.euclidean(generator.x[first_city], 
                                                        generator.x[second_city])
            selected_cities[[first_city, second_city]] = 1
            other_cities = np.delete(np.arange(N), [first_city, second_city])
            for c in other_cities:
                d_to_first = scipy.spatial.distance.euclidean(generator.x[first_city], 
                                                        generator.x[c])
                d_to_second = scipy.spatial.distance.euclidean(generator.x[second_city], 
                                                        generator.x[c])
                if (d_to_first <= max_dist) & (d_to_second <= max_dist):
                    selected_cities[c] = 1
            f = objective_function_simple(N, l, generator, selected_cities, 
                                      pairwise_distances=pairwise_distances)
            
            if f < min_f:
                solution = selected_cities
                min_f = f
    return solution, min_f
            