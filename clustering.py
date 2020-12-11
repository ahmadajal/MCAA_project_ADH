import matplotlib.pyplot as plt
from matplotlib.path import Path

import numpy as np

import scipy
import scipy as sp
import scipy.stats as st
from scipy.spatial import ConvexHull

import tqdm
import tqdm.notebook


def plotResult(data,duration,selected_cities_n,selected_cities_n_convex,loss_values,loss_value_convex,num_cities_per_step):
    print("d= %s seconds" % duration)
    if loss_value_convex is not None:
        print('Final loss '+ str(loss_values[-2])) #be careful, at the end we want to return '-final loss'
        print('Final loss with Convex Hull '+ str(loss_value_convex))
    else:
        print('Final loss '+ str(loss_values[-1]))
    #print(selected_cities_n)

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    #fig.suptitle('Results')
    axes[0].plot(loss_values)
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Iterations')
    axes[0].set_title('Loss Evolution')
    m = selected_cities_n == 1
    axes[1].scatter(data.x[:, 0], data.x[:, 1],label='Non selected Cities')
    axes[1].scatter(data.x[m, 0], data.x[m, 1], c='r',label='Selected cities')
    if selected_cities_n_convex is not None:
        mbis = (selected_cities_n_convex==1) & (selected_cities_n==0)
        mter = (selected_cities_n_convex==0) & (selected_cities_n==1)
        axes[1].scatter(data.x[mbis, 0], data.x[mbis, 1], c='g',label='Added cities (Convex Hull)')
        axes[1].scatter(data.x[mter, 0], data.x[mter, 1], c='y',label='Selected Cities not in Convex Hull')
    axes[1].set_title('Selected cities')
    box = axes[1].get_position()
    axes[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if num_cities_per_step is not None:
        plt.figure(figsize=(4,2))
        plt.plot(np.arange(n_iter), num_cities_per_step)
        plt.title("#selected cities in each step")