import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def part1(X, y, lambd):
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    
    X_transpose = np.transpose(X)
    
    X_transpose_X = np.dot(X_transpose, X)
    
    X_transpose_y = np.dot(X_transpose, y)
    
    identity_matrix = np.eye(X_transpose_X.shape[0])
    
      
    wRR = np.dot(np.linalg.inv(lambd*identity_matrix + X_transpose_X), X_transpose_y).tolist()
    
    return wRR

wRR = part1(X_train, y_train, lambda_input)  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2(X, lambd, sigma2, D):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    
    # Calculate prior covariance
    X_transpose = np.transpose(X)
    X_transpose_X = np.dot(X_transpose, X)
    identity_matrix = np.eye(X_transpose_X.shape[0])
    
    # Calculate prior covariance
    covariance = np.linalg.inv(lambd*identity_matrix + (X_transpose_X)/sigma2)
    
    # Initialize list to hold index of minimum entropy 
    index = []
    D_temp = D
    # Loop through to identify top ten indexes
    for count in range(10):
        # Calculate entropy of all x0 in D_temp
        entropy = []
        
        # Loop through all x0 to identify index with minimum entropy
        for x0 in D_temp:
            x0_transpose = np.transpose(x0)
            entropy.append(sigma2 + np.dot(np.dot(x0_transpose, covariance), x0))
        arg_max = np.argmax(np.asarray(entropy))
        x0_max = D_temp[arg_max]
        
        # Remove minimum entropy x0 from D_temp for next iteration
        D_temp = np.delete(D_temp, arg_max, 0)
        
        # Store index of minimum entropy x0 from D 
        index.append(np.where(np.all(D==x0_max, axis=1))[0][0] + 1)

        # Calculate posterior covariance (prior for next iteration)
        x0_max_transpose = np.transpose(x0_max)
        x0_max_x0_max_transpose = np.dot(x0_max, x0_max_transpose)
        covariance = np.linalg.inv(np.linalg.inv(covariance) + x0_max_x0_max_transpose/sigma2)
    return index

active = part2(X_train, lambda_input, sigma2_input, X_test)  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + 
str(int(sigma2_input)) + ".csv", [active]       , delimiter=",") # write output to file