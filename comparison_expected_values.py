import scipy.stats as st
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle

import tqdm 
import tqdm.notebook
import scipy 
import pandas as pd
# temp: functions and more...
from util import G1, G2
import multiprocessing
import baseline
import smooth
import convexhull
import clustering
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_cities', type=int, default=1000, help='number of cities')
argparser.add_argument('--n_iter', type=int, default=10000, help='number of iterations')
argparser.add_argument('--num_runs', type=int, default=20, help='number of runs per setup')
args = argparser.parse_args()
print(args)
# beta scheduler
def beta(i, n_iter):
    if i < n_iter * (1/5) :
        return 1
    elif i < n_iter * (2/5):
        return 5
    elif i < n_iter * (3/5):
        return 10
    elif i < n_iter * (4/5):
        return 20
    else: 
        return 50
    
def effect_of_lambda(n_iter, beta, generator, N, lambda_range, num_runs, mutation_strategy):
    strategy_names = {'Baseline':baseline, 'Convex hull': convexhull, 
                'Clustering': clustering, 'Smooth': smooth}
    # dictionary to save the results
    results_f = {l: [] for l in lambda_range}
    results_num_cities = {l: [] for l in lambda_range}
    for i in range(num_runs):
        cities = generator(N)
        for l in lambda_range:
            if mutation_strategy != "Clustering": 
                curr_selected, curr_loss_values, number_of_selected_cities = strategy_names[mutation_strategy].optimize(
                        cities, l, beta=beta, n_iter=n_iter, verbose=False)
            else: # clustering 
                curr_selected, curr_loss_values, number_of_selected_cities = strategy_names[mutation_strategy].optimize(
                        cities, l, beta=10, n_iter=n_iter, verbose=False)
            # save the result
            results_f[l].append(curr_loss_values[-1])
            results_num_cities[l].append(number_of_selected_cities[-1])
    return results_num_cities, results_f

N = args.num_cities
num_iter = args.n_iter
num_runs = args.num_runs
lambda_range1 = np.linspace(0, 1, 11)
lambda_range2 =  np.linspace(0, 2, 21)
init_prob=0.0

r_G1_num_cities={"Baseline": [], "Convex hull": [], "Clustering": [], "Smooth": []}
r_G1_f={"Baseline": [], "Convex hull": [], "Clustering": [], "Smooth": []}

r_G2_num_cities={"Baseline": [], "Convex hull": [], "Clustering": [], "Smooth": []}
r_G2_f={"Baseline": [], "Convex hull": [], "Clustering": [], "Smooth": []}
for i in ["Baseline", "Convex hull", "Clustering", "Smooth"]:
    r_G1_num_cities[i], r_G1_f[i] =  effect_of_lambda(num_iter, beta, G1, N, lambda_range1, num_runs, mutation_strategy=i)

    r_G2_num_cities[i], r_G2_f[i] =  effect_of_lambda(num_iter, beta, G2, N, lambda_range2, num_runs, mutation_strategy=i)
    print("strategy {} finished".format(i))
# save
with open("results_cities_G1_{}.pkl".format(N), "wb") as f:
    pickle.dump(r_G1_num_cities, f)
with open("results_f_G1_{}.pkl".format(N), "wb") as f:
    pickle.dump(r_G1_f, f)
with open("results_cities_G2_{}.pkl".format(N), "wb") as f:
    pickle.dump(r_G2_num_cities, f)
with open("results_f_G2_{}.pkl".format(N), "wb") as f:
    pickle.dump(r_G2_f, f)

    
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

fig.text(0.4, 1, r"Effect of $\lambda$. Results are for {} runs".format(num_runs), fontsize=12)
for i in ["Baseline", "Convex hull", "Clustering", "Smooth"]:
    ax[0][0].plot(lambda_range1, [np.mean(i) for i in r_G1_num_cities[i].values()], label="strategy "+i)
    ax[0][0].set_title(r"$\mathbb{E}_{\mathcal{G}} [|\mathcal{S}^{\ast}(\lambda)|]$ for uniform generator $\mathcal{G}_1$")
    ax[0][0].legend()

for i in ["Baseline", "Convex hull", "Clustering", "Smooth"]:
    ax[0][1].plot(lambda_range2, [np.mean(i) for i in r_G2_num_cities[i].values()], label="strategy "+i)
    ax[0][1].set_title(r"$\mathbb{E}_{\mathcal{G}} [|\mathcal{S}^{\ast}(\lambda)|]$ for log_normal generator $\mathcal{G}_2$")
    ax[0][1].legend()

for i in ["Baseline", "Convex hull", "Clustering", "Smooth"]:
    ax[1][0].plot(lambda_range1, [np.mean(i) for i in r_G1_f[i].values()], label="strategy "+i)
    ax[1][0].set_title(r"$\mathbb{E}_{\mathcal{G}} [f(\lambda, \mathcal{S}^{\ast}(\lambda))]$ for uniform generator $\mathcal{G}_1$")
    ax[1][0].legend()
    
for i in ["Baseline", "Convex hull", "Clustering", "Smooth"]:
    ax[1][1].plot(lambda_range2, [np.mean(i) for i in r_G2_f[i].values()], label="strategy "+i)
    ax[1][1].set_title(r"$\mathbb{E}_{\mathcal{G}} [f(\lambda, \mathcal{S}^{\ast}(\lambda))]$ for log-normal generator $\mathcal{G}_2$")
    ax[1][1].legend()
    
ax[1][0].set_xlabel(r"$\lambda$")
ax[1][1].set_xlabel(r"$\lambda$")
ax[0][0].set_ylabel(r"$\mathbb{E}_{\mathcal{G}} [|\mathcal{S}^{\ast}(\lambda)|]$", fontsize=12)
ax[1][0].set_ylabel(r"$\mathbb{E}_{\mathcal{G}} [f(\lambda, \mathcal{S}^{\ast}(\lambda))]$", fontsize=12)
fig.savefig("Expectation_f_num_cities_{}.pdf".format(N), dpi=1200);
