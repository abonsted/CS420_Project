import numpy as np
import pandas as pd

#Global vals
m = 5 #These will be the "m" fundamental memories imprinted
num_neurons = 50 

def sign(x):
    if(x >= 0):
        return 1
    else:
        return -1
    
def imprintPatterns(patterns):
    W = np.zeros((num_neurons, num_neurons))
    for p in patterns:
        i_vec = np.array([p]).transpose()
        j_vec = np.array([p])

        weight_matrix = np.dot(i_vec, j_vec)

        weight_matrix = weight_matrix - np.identity(num_neurons)

        W += weight_matrix
    return W
    
#Loop through increasing number of patterns (1->20)
for i in range(1, m):
    #Make patterns (randomize or images)
    patterns = np.random.choice([-1, 1], (i, num_neurons))

    #Imprint those patterns and create weight matrix
    
    weights = imprintPatterns(patterns)

    #Recall process, async vs sync or pick one

    break

"""
Things to look into
 - async vs sync recalling - thurs
 - Images rather than random
 - Noise?
 - Imprinted patterns
"""