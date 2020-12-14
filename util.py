import matplotlib.pyplot as plt
from matplotlib.path import Path

import numpy as np

import scipy
import scipy as sp
import scipy.stats as st
from scipy.spatial import ConvexHull


def maximum_dist(cities_pos):
    if cities_pos.shape[0] > 2:
        max_distance = np.max(scipy.spatial.distance.pdist(
            cities_pos[ConvexHull(cities_pos).vertices, :], 'sqeuclidean'))
    elif cities_pos.shape[0] == 2:
        max_distance = np.max(scipy.spatial.distance.pdist(cities_pos, 'sqeuclidean'))
    else:
        max_distance = 0.0
    return max_distance


def objective_function(l, cities, selected_cities):
    if len(selected_cities) == 0:
        return np.inf
    N = cities.x.shape[0]
    selected_cities_pos = cities.x[selected_cities == 1, :]
    max_distance = maximum_dist(selected_cities_pos)
    return -np.sum(cities.v[selected_cities == 1]) + l * N * max_distance * np.pi / 4


def add_points_in_convex_hull(l, cities, selected_cities):
    """l: lambda, cities: cities of the current problem selected_cities: 1 hot array of selected cities"""
    N = cities.x.shape[0]
    selected_cities_pos = cities.x[selected_cities == 1, :]
    convex_hull = selected_cities_pos[ConvexHull(selected_cities_pos).vertices, :]
    max_distance = np.max(scipy.spatial.distance.pdist(convex_hull, 'sqeuclidean'))
    hull_path = Path(convex_hull)
    is_in_hv = hull_path.contains_points(cities.x)
    is_in_hv[selected_cities == 1] = 1
    loss = -np.sum(cities.v[is_in_hv == 1]) + l * N * max_distance * np.pi / 4
    return is_in_hv, loss
