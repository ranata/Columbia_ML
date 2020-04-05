import numpy as np
import pandas as pd
import scipy as sp
import sys

X = np.genfromtxt("X.csv", delimiter = ",")

def KMeans_one_step(X, mu):
    C = []
    ctr = 1
    # Assignment step - assign each x to the nearest centroid. This means find ci for each xi.
    for x in X:
        l2 = []
        for mu_k in mu:
            dist = np.linalg.norm(x - mu_k)**2
            l2.append(dist)
        C.append(np.argmin(l2))
    K, n_K = np.unique(C, return_counts=True)
    
    # Update step - Recalculate centroid for each cluster based on xi assignment to the cluster

    _mu = []
    for k, n_k in zip(K, n_K):
        mu_k = np.dot((C == k), X)/n_k
        _mu.append(mu_k)
    return _mu

def KMeans(data):
	#perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively

    # Initialize the centroids randomly
    mu = np.random.rand(5, data.shape[1])
    
    # Calculate centroid for each cluster for 10 iterations
    for i in range(10):
        mu = KMeans_one_step(X, mu)

        filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
        np.savetxt(filename, mu, delimiter=",")
        
  
#def EMGMM(data):

#	filename = "pi-" + str(i+1) + ".csv" 
#	np.savetxt(filename, pi, delimiter=",") 
#	filename = "mu-" + str(i+1) + ".csv"
#	np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
#    
#    for j in range(k): #k is the number of clusters 
#        filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
#        np.savetxt(filename, sigma[j], delimiter=",")
#

if __name__ == '__main__':

    data = np.genfromtxt("X.csv", delimiter = ",")
    
    KMeans(data)
    
        # EMGMM(data)
