import numpy as np
import pandas as pd
import scipy as sp
import sys

X = np.genfromtxt(sys.argv[1], delimiter = ",")

def weighted_matrix(weight, matrix):
    weighted_matrix = []
    for i in range (weight.shape[0]):
        weighted_matrix.append(weight[i]*matrix)
    return np.array(weighted_matrix)

def calc_mean(X, phi, n_k):
    phi_x = 0
    for i in range(X.shape[0]):
        phi_x = phi_x + weighted_matrix(phi[i], X[i])
    return list(phi_x/np.array(n_k.reshape(len(n_k), 1)))

def calc_covariance(X, phi, mu, n_k):
    phi_covariance = []
    for k in (range(len(mu))):
        cov_k = 0
        for i in range(X.shape[0]):
            mean_dev = np.array((X[i] - mu[k]).reshape(len(X[i] - mu[k]), 1))
            cov_k = cov_k + np.dot(mean_dev, mean_dev.T)*phi[i][k]/n_k[k]
        phi_covariance.append(cov_k)
    return phi_covariance

def KMeans_one_step(X, mu):
    C = []
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

def EMGMM_one_step(X, pi, mu, sigma):
    # E-Step - calculate weightage (phi) on each cluster using Gaussian probablity distribution
    phi = []
    posterior = []
    for x in X:
        posterior_k = []
        for k in range(len(pi)):
            mean_dev = x - mu[k]
            coefficient = 1/np.sqrt(((2*np.math.pi)**X.shape[1])*np.linalg.det(sigma[k]))
            mahalanobis_distance = np.dot(np.dot(mean_dev.T, np.linalg.inv(sigma[k])), mean_dev)

            pdf_k = coefficient*np.exp(-0.5*mahalanobis_distance)
            posterior_k.append(pdf_k*pi[k])
        posterior.append(posterior_k)
        phi.append(posterior_k/np.asarray(posterior_k).sum())
    
    # M-Step - update pi, mu and sigma for each cluster
    n_k = np.array(phi).sum(axis=0)

    _pi = n_k/X.shape[0]
    _mu = calc_mean(X, np.array(phi), n_k)
    _sigma = calc_covariance(X, phi, mu, n_k)
    
    return (_pi, _mu, _sigma)


def KMeans(data):
	#perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively

    # Initialize the centroids randomly
    mu = np.random.rand(5, data.shape[1])
    
    # Calculate centroid for each cluster for 10 iterations
    for i in range(10):
        mu = KMeans_one_step(X, mu)

        filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
        np.savetxt(filename, mu, delimiter=",")
        
  
def EMGMM(data):
    # Initialize parameters
    np.random.seed(6962)
    k = 5 # No. of clusters
    pi = list(np.random.uniform(size=k))
    mu = list(np.random.rand(k, X.shape[1]))
    sigma = list([np.identity(X.shape[1])]*k)
    
    
    # Calculate prior, centroid, covariance for each cluster for 10 iterations
    for i in range(10):
        (pi, mu, sigma) = EMGMM_one_step(X, pi, mu, sigma)
        filename = "pi-" + str(i+1) + ".csv" 
        np.savetxt(filename, pi, delimiter=",") 
        filename = "mu-" + str(i+1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
    
        for j in range(k): #k is the number of clusters 
            filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, sigma[j], delimiter=",")


if __name__ == '__main__':

    data = np.genfromtxt(sys.argv[1], delimiter = ",")
    
    KMeans(data)
    
    EMGMM(data)
