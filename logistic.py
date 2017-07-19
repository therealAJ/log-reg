'''
Logistic Regression from scratch

Thanks to Andrew Ng's Stanford Coursera notes for providing
references to needed equations
'''

import math

def sigmoid(z):
    return 1 / (1 + math.exp(-1 * z))

'''
Can be calculated using either vectorized or iterative versions
vectorized = h(x) = 1 / 1 + e^ - (theta_transpose * X)
we can calculate theta_transpose * x by calculating the dot product
(multiplying each column in the row vector theta_tranpose by the matching
row in the column vector X and then sum up the products) 
'''
def hypothesis(theta, x):
    z = 0
    for i in range(len(theta)):
        z += theta[i] * x[i]
    return sigmoid(z)


