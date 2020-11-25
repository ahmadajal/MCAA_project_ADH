import scipy.stats as st
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
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
        self.x = st.uniform().rvs((self.N,2))
        self.v = st.uniform().rvs((self.N,))

class G2(DatasetGenerator):
    def refresh(self):
        self.x = st.uniform().rvs((self.N,2))
        self.v = np.exp(st.norm(-0.85, 1.3).rvs((self.N,)))
        
# an optimal beta as said in the course, could be interesting to use it
def computeBeta(N0, N1, f0, f1, epsilon):
    return log(N1 / (epsilon * N0)) / (f1 - f0)

def objectiveFunction(N, l, citiesV, selectedCities, pairwise_distances):
    # Compute maximum area
    max_area = np.pi / 4 * np.max(np.outer(selectedCities, selectedCities) * pairwise_distances)

    # Compute final loss
    f = np.sum(selectedCities * citiesV) - l * N * max_area
    return -f

def acceptancePb(selectedCities_i,selectedCities_j,beta,N,l,citiesV, pairwise_distances):
    fi = objectiveFunction(N, l, citiesV, selectedCities_i, pairwise_distances)
    fj = objectiveFunction(N, l, citiesV, selectedCities_j, pairwise_distances)
    result = np.exp(-beta * (fj - fi))
    return min(1, result)

def pbij(selectedCities_i, selectedCities_j, beta, N, l, citiesV):
    # not used in the algorithm, could be useful to plot and compare statistics
    if selectedCities_i == selectedCities_j:
        s = 0
        for k in range (N):
            selectedCities_k=np.copy(selectedCities_i)
            selectedCities_k[k]=1-selectedCities_i[k]
            a_ik=acceptancePb(selectedCities_i,selectedCities_k,beta,n,l,citiesV)
            phi_ik=(1/2)*(1/N)
            s+=phi_ik*a_ik
        return 1-s
    else:
        a_ij=acceptancePb(selectedCities_i,selectedCities_j,beta,n,l,citiesV)
        phi_ij=(1/2)*(1/N) #pb of chosing a city*pb choosing if 0 or 1
        return a_ij*phi_ij
    
# Step 2,3,4
def step(N, citiesX, citiesV, selectedCities_i, beta, l, pairwise_distances):
    k = np.random.randint(0, N);
    if np.random.rand() < 0.5: # Remove a city
        if selectedCities_i[k] == 0: # same state then before
            return selectedCities_i # do nothing, it is accepted
        else:
            selectedCities_k = np.copy(selectedCities_i)
            selectedCities_k[k] = 0 # city removed from set
            a_ik = acceptancePb(selectedCities_i, selectedCities_k, beta, N, l, citiesV, pairwise_distances) 
            if np.random.rand() < a_ik:
                return selectedCities_k #accepted!
            else:
                return selectedCities_i #refused
    else: # Add a city
        if selectedCities_i[k] == 1: # do nothing, city already in set
            return selectedCities_i
        else:
            selectedCities_k = np.copy(selectedCities_i)
            selectedCities_k[k] = 1 # add city to set
            #could of course be computed in a smarter way
            a_ik = acceptancePb(selectedCities_i, selectedCities_k, beta, N, l, citiesV, pairwise_distances)
            if np.random.rand() < a_ik:
                return selectedCities_k #city added!
            else:
                return selectedCities_i #refused

            
