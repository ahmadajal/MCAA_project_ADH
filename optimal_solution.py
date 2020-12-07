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
    # better solution
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
            dist_to_fisrt = np.sqrt(((generator.x - generator.x[first_city])**2).sum(axis=1))
            dist_to_second = np.sqrt(((generator.x - generator.x[second_city])**2).sum(axis=1))
            # filter the cities whose distance to first is less than max_dist
            cities_to_first = np.where(dist_to_fisrt <= max_dist)[0]
            # filter the cities whose distance to second is less than max_dist
            cities_to_second = np.where(dist_to_second <= max_dist)[0]
            # selected cities is the intersection of the above sets
            selected_cities_idx = np.intersect1d(cities_to_first, cities_to_second)
            # add the new cities to the list
            selected_cities[selected_cities_idx] = 1
            f = objective_function_simple(N, l, generator, selected_cities, 
                                      pairwise_distances=pairwise_distances)
            
            if f < min_f:
                solution = selected_cities
                min_f = f
    return solution, min_f
            