from __future__ import division
import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

## can make more functions if required


def pluginClassifier(X_train, y_train, X_test):

    Y, n_y = np.unique(y_train, return_counts=True)
    pi_hat = []
    mu_hat = []
    sigma_hat = []
    
    for y, count_y in zip(Y, n_y):
        pi_hat_y = count_y/np.sum(n_y)
        pi_hat.append(pi_hat_y)
        mu_hat_y = np.dot((y_train == y), X_train)/count_y
        mu_hat.append(mu_hat_y)
        X_train_y = X_train[y_train.astype(int) == np.int(y), :]
        
        x_mu_hat_dev_y = X_train_y - mu_hat_y
            
        sigma_hat_y = np.dot(x_mu_hat_dev_y.T, x_mu_hat_dev_y)/(count_y)
        
        sigma_hat.append(sigma_hat_y)
    
    
    # predicting for X_test
    y_test = []
    
    for x in X_test:
    
        f_x_hat = []
        for y in Y:
            idx = np.int(y)
            x_mu_hat_dev_y = x - mu_hat[idx]
    
            coefficient = pi_hat[idx]/np.sqrt(np.linalg.det(sigma_hat[idx]))
            
            power_value = -(1/2)*(x_mu_hat_dev_y.T@np.linalg.inv(sigma_hat[idx])@x_mu_hat_dev_y)
     
            f_x_hat_y = coefficient*np.exp(power_value)
            f_x_hat.append(f_x_hat_y)
    
        y_test.append(f_x_hat/np.asarray(f_x_hat).sum())
    
    return np.asarray(y_test)

     

final_outputs = pluginClassifier(X_train, y_train, X_test) # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file