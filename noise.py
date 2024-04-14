import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Global vals
m = 10 #These will be the "m" fundamental memories imprinted
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

def async_recall(probe, weights, patterns):
    min_energy = 1000
    min_probe = []
    iter = 0
    while iter < 1000:
        neuron_index = np.random.randint(0, len(probe))
        probe[neuron_index] = sign(np.dot(weights[neuron_index], probe))

        #Check energy threshold
        energy = calculate_energy(probe, weights)
        if(energy < min_energy):
            min_energy = energy
            min_probe = probe

            for p in patterns:
                equal = np.array_equal(probe, p)
                if equal:
                    return min_probe, min_energy, iter
        iter += 1

    return min_probe, min_energy, iter
    
def calculate_energy(probe, weights):
    energy = 0.0
    for i in range(num_neurons):
        for j in range(num_neurons):
            if(i != j):
                energy += weights[i,j]*probe[i]*probe[j]
    energy *= -0.5
    return energy


def hop_net(noise_lvl):
    #Loop through increasing number of patterns (1->20)
    recall_count = np.zeros(m+1)
    for i in range(1, m):
        #Make patterns (randomize or images)
        patterns = np.random.choice([-1, 1], (i, num_neurons))

        #Imprint 'i' patterns and create weight matrix
        weights = imprintPatterns(patterns)

        #Asynchronous recall process
        for num, p in enumerate(patterns): # testing j number of probes with given weight matrix
            #Create noisy probe from pattern p (alr imprinted)
            noisy = p.copy()
            noisy_ind = np.random.choice(num_neurons, (int)(num_neurons * noise_lvl), replace=False)
            for ind in noisy_ind:
                noisy[ind] *= -1
            probe = noisy
            new_probe, min_e, iter = async_recall(probe, weights, patterns)

            if(iter < 1000):
                recall_count[i] += 1
        recall_count[i] /= i

    print(noise_lvl)
    print(recall_count[1:-1])

#MAIN here
# noise_lvls = np.arange(0.1, 0.7, 0.1)
# print(noise_lvls)
# for i in noise_lvls:
#     hop_net(i)

hop_net(0.2)