'''
Logistic Regression from scratch

Thanks to Andrew Ng's Stanford Coursera notes for providing
references to needed equations
'''

import math

def sigmoid(z):
    return float(1 / float((1 + math.exp(-1 * z))))

'''
The hypothesis can be calculated using either vectorized or iterative versions
vectorized = h(x) = 1 / 1 + e ^ -1 * (theta_transpose * X)
we can calculate theta_transpose * x by calculating the dot product
(multiplying each column in the row vector theta_tranpose by the matching
row in the column vector x and then sum up the products) 
'''
def hypothesis(theta,x):
    z = 0
    for i in range(len(theta)):
        z += theta[i] * x[i]
    return sigmoid(z)

def cost_function(X,y,theta,m):
    sumError = 0
    for i in range(m):
        error = 0
        x_i = X[i]
        y_i = y[i]
        h_i = hypothesis(theta,x_i)
        error = y_i * math.log(h_i) + (1 - y_i) * math.log(1 - h_i)
        sumError += error
    const = -1/m
    cost = const * sumError
    return cost

def cfd(X,y,theta,j,m):
    sumError = 0
    for i in range(m):
        x_i = X[i]
        x_ij = x_i[j]
        h_i = hypothesis(theta, x_i)
        error = (h_i - y[i]) * x_ij
        sumError += error
    const = 1/m
    cost = const * sumError
    return cost

def gradient_descent(X,y,theta,alpha,m):
    opt_theta = []
    for j in range(len(theta)):
        cost = cfd(X,y,theta,j,m)
        updated_theta = theta[j] - (alpha * cost)
        opt_theta.append(updated_theta)
    return opt_theta

def Logistic_Regression(X, y, alpha, theta, iterations):
    m = len(y)
    for i in range(iterations):
        opt_theta = gradient_descent(X,y,theta,alpha,m)
        theta = opt_theta
    return theta