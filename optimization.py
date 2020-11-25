import matplotlib.pyplot as plt

import numpy as np

import scipy
import scipy as sp
import scipy.stats as st

import tqdm
import tqdm.notebook


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
    max_area = np.pi / 4 * np.max(np.outer(selected_cities, selected_cities) * pairwise_distances)
    f = np.sum(selected_cities * cities.v) - l * N * max_area
    return -f


def step(N, cities, selected_cities_i, current_loss_value, beta, l, pairwise_distances, mutation_strategy=0):
    k = np.random.randint(0, N)
    remove_city = np.random.rand() < 0.5
    if (mutation_strategy == 0) and ((remove_city and selected_cities_i[k] == 0) or (not remove_city and selected_cities_i[k] == 1)):
        return (selected_cities_i, current_loss_value)  # do nothing
    else:
        selected_cities_k = np.copy(selected_cities_i)
        selected_cities_k[k] = 1 - selected_cities_k[k]
        new_loss_value = objective_function(N, l, cities, selected_cities_k, pairwise_distances)
        a_ik = min(1, np.exp(-beta * (new_loss_value - current_loss_value)))
        return (selected_cities_k, new_loss_value) if np.random.rand() < a_ik else (selected_cities_i, current_loss_value)


def optimize(cities, l, beta=100, n_iter=20000, mutation_strategy=0, initial_selection_probability=0.5):
    """mutation_strategy = 0: Original mutation proposed by Heloise
       mutation_strategy = 1: Simple strategy which just randomly tries to flip cities
       initial_selection_probability: Probablity at which a city initially is selected (0.5: every city can be selected with 50% chance)"""

    N = cities.x.shape[0]
    selected_cities = (np.random.rand(N) <= initial_selection_probability).astype(np.int32)
    pairwise_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(cities.x, 'sqeuclidean'))
    fs = np.zeros(n_iter)
    current_loss_value = objective_function(N, l, cities, selected_cities, pairwise_distances)
    for m in tqdm.notebook.tqdm(range(n_iter)):
        fs[m] = current_loss_value
        selected_cities, current_loss_value = step(
            N, cities, selected_cities, current_loss_value, beta, l, pairwise_distances,
            mutation_strategy=mutation_strategy)
    return selected_cities, fs


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
