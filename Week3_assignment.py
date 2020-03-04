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
    
    X_transpose_y = np.dot(X_transpose, y)
    
    identity_matrix = np.eye(X_transpose_X.shape[0])
    
      
    wRR = np.dot(np.linalg.inv(lambd*identity_matrix + X_transpose_X), X_transpose_y).tolist()
    
    return wRR
    pass

wRR = part1(X_train, y_train, lambda_input)  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2(X, lambd, sigma2):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    
    X_transpose_X = np.dot(X_transpose, X)
    
    identity_matrix = np.eye(X_transpose_X.shape[0])
    
    covar_sigma = np.linalg.inv(lambd*identity_matrix + (X_transpose_X)/sigma2)
    pass

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + 
           str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file