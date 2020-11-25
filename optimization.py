import scipy.stats as st
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import tqdm
import tqdm.notebook
import scipy


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


class G2(DatasetGenerator):
    def refresh(self):
        self.x = st.uniform().rvs((self.N, 2))
        self.v = np.exp(st.norm(-0.85, 1.3).rvs((self.N,)))


# an optimal beta as said in the course, could be interesting to use it
def compute_beta(N0, N1, f0, f1, epsilon):
    return np.log(N1 / (epsilon * N0)) / (f1 - f0)


def objective_function(N, l, cities, selected_cities, pairwise_distances):
    # Compute maximum area
    max_area = np.pi / 4 * np.max(np.outer(selected_cities, selected_cities) * pairwise_distances)
    # Compute final loss
    f = np.sum(selected_cities * cities.v) - l * N * max_area
    return -f


def acceptance_pb(selected_cities_i, selected_cities_j, beta, N, l, cities, pairwise_distances):
    fi = objective_function(N, l, cities, selected_cities_i, pairwise_distances)
    fj = objective_function(N, l, cities, selected_cities_j, pairwise_distances)
    result = np.exp(-beta * (fj - fi))
    return min(1, result)


def step(N, cities, selected_cities_i, beta, l, pairwise_distances):
    k = np.random.randint(0, N)
    if np.random.rand() < 0.5:  # Remove a city
        if selected_cities_i[k] == 0:
            return selected_cities_i  # same state then before, do nothing
        else:
            selected_cities_k = np.copy(selected_cities_i)
            selected_cities_k[k] = 0  # city removed from set
            a_ik = acceptance_pb(selected_cities_i, selected_cities_k, beta, N, l, cities, pairwise_distances)
            return selected_cities_k if np.random.rand() < a_ik else selected_cities_i
    else:  # Add a city
        if selected_cities_i[k] == 1:  # do nothing, city already in set
            return selected_cities_i
        else:
            selected_cities_k = np.copy(selected_cities_i)
            selected_cities_k[k] = 1  # add city to set
            a_ik = acceptance_pb(selected_cities_i, selected_cities_k, beta, N, l, cities, pairwise_distances)
            return selected_cities_k if np.random.rand() < a_ik else selected_cities_i


def optimize(cities, l, beta=100, n_iter=20000):
    N = cities.x.shape[0]
    selected_cities = np.random.randint(2, size=N)
    pairwise_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(cities.x, 'sqeuclidean'))
    fs = np.zeros(n_iter)
    fs[0] = objective_function(N, l, cities, selected_cities, pairwise_distances)
    for m in tqdm.notebook.tqdm(range(n_iter)):
        fs[m] = objective_function(N, l, cities, selected_cities, pairwise_distances)
        selected_cities = step(N, cities, selected_cities, beta, l, pairwise_distances)
    return selected_cities, fs


# Old code
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
