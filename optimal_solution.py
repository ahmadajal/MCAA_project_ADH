import numpy as np
import matplotlib.pyplot as plt

import tqdm
import tqdm.notebook

from optimization import *

import time

import scipy
from scipy import spatial

import itertools

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


def opt_solution_circle(N, l, generator, verbose=False):
    # Alternative solution: Only add cities in the circle spanned by the two extreme points
    min_f = np.inf
    pairwise_distances = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(generator.x, 'sqeuclidean'))
    it = tqdm.notebook.tqdm(range(N-1)) if verbose else range(N-1)
    for first_city in it:
        for second_city in range(first_city, N):
            selected_cities = np.zeros(N, dtype=int)
            max_dist = np.sum((generator.x[first_city] - generator.x[second_city]) ** 2)
            selected_cities[[first_city, second_city]] = 1
            mid_point = 0.5 * (generator.x[first_city] + generator.x[second_city])
            selected_cities[((generator.x - mid_point) ** 2).sum(axis=1) <= max_dist / 4] = 1
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
            max_dist = np.sum((generator.x[first_city] - generator.x[second_city]) ** 2)
            selected_cities[[first_city, second_city]] = 1
            dist_to_first = ((generator.x - generator.x[first_city]) ** 2).sum(axis=1)
            dist_to_second = ((generator.x - generator.x[second_city]) ** 2).sum(axis=1)
            selected_cities[(dist_to_first <= max_dist) * (dist_to_second <= max_dist)] = 1
            f = objective_function_simple(N, l, generator, selected_cities,
                                          pairwise_distances=pairwise_distances)
            if f < min_f:
                solution = selected_cities
                min_f = f
    return solution, min_f


def opt_solution_kdtree(N, l, generator, verbose=False):
    """Optimal solution using a KD tree acceleration data structure, not really faster than without it though"""
    min_f = np.inf
    pairwise_distances = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(generator.x, 'sqeuclidean'))
    it = tqdm.notebook.tqdm(range(N-1)) if verbose else range(N-1)
    kd_tree = spatial.cKDTree(generator.x)
    for first_city in it:
        for second_city in range(first_city, N):
            selected_cities = np.zeros(N, dtype=int)
            max_dist = np.sqrt(np.sum((generator.x[first_city] - generator.x[second_city]) ** 2))
            selected_cities[[first_city, second_city]] = 1
            idx0 = kd_tree.query_ball_point(generator.x[first_city], max_dist + 1e-6)
            idx1 = kd_tree.query_ball_point(generator.x[second_city], max_dist + 1e-6)
            selected_cities_idx = np.intersect1d(idx0, idx1)
            selected_cities[selected_cities_idx] = 1
            f = objective_function_simple(N, l, generator, selected_cities,
                                          pairwise_distances=pairwise_distances)
            if f < min_f:
                solution = selected_cities
                min_f = f
    return solution, min_f