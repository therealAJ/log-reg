'''
Logistic Regression from scratch

Thanks to Andrew Ng's Stanford Coursera notes for providing
references to needed equations
'''

import math

def Sigmoid(z):
    return 1 / (1 + math.exp(-1 * z))

