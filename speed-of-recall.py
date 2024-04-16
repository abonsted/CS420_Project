# Modifiations of main.py written by Andrew Bonsted
# In preparations for the week of 4/15/24
# Measuring the speed of recall (number of asynchronous steps needed vs. number of patterns imprinted)


from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from sys import exit
from io import BytesIO
from PIL import Image as Image

# make a seed for more consistent results
# seed = 1794
# np.random.seed(seed)


#Global vals
m = 20 #These will be the "m" fundamental memories imprinted
num_neurons = 22 * 22
iter_threshold = 100

# convert png image from PIL to usable format
def png_to_binary_img(infile_path):
    # img = Image.open(png_path)
    
    # grayscale_img = img.convert("L")
    
    # bin_threshold = 128
    # binary_img = grayscale_img.point(lambda x: -1 if x < bin_threshold else 1)
    # print(grayscale_img)
    # bin_arr = np.array(binary_img)
    img_arr = np.array(Image.open(infile_path).convert("L")).astype('int32')
    all_cols = img_arr[:, :]
    all_cols[all_cols == 0] = -1
    all_cols[all_cols > 0] = 1
    img_arr = img_arr[3:img_arr.shape[0] - 3, 3:img_arr.shape[1] - 3]
    #print(bin_arr)
    #print(desquarify(bin_arr))
    # plt.imshow(bin_arr, cmap="binary")
    # plt.show()
    
    return img_arr

# convert square 2D array into single NxN 1D array
def desquarify(pattern):
    return pattern.reshape((pattern.shape[0] * pattern.shape[1]))

# Converted 1D array of NxN elements into a 2D N x N matrix
def squarify(pattern):
    return pattern.reshape( int(sqrt(pattern.shape[0])), int(sqrt(pattern.shape[0])))

# sign func
def sign(x):
    if(x >= 0):
        return 1
    else:
        return -1

# Imprint patterns to weight matrix
def imprintPatterns(patterns):
    
    W = np.zeros((num_neurons, num_neurons))
    for p in patterns:
        i_vec = np.array([p]).transpose()
        j_vec = np.array([p])

        weight_matrix = np.dot(i_vec, j_vec)

        weight_matrix = weight_matrix - np.identity(num_neurons)

        W += weight_matrix
    return W

# Reconstruct image with async recall and return info about the closest pattern
# retrieved 
def async_recall(probe, weights, patterns):
    min_energy = 1000
    min_probe = []
    iter = 0
    while iter < iter_threshold:
        
        neuron_index = np.random.randint(0, len(probe))
        before = probe[neuron_index]
        after = sign(np.dot(weights[neuron_index], probe))
        probe[neuron_index] = after
        
        # Recompute energy if a state changed
        #Check energy threshold
        energy = calculate_energy(probe, weights)
        if(energy < min_energy):
            print("NEW MIN: ", energy, " on iteration: ", iter)
            min_energy = energy
            min_probe = probe

            for p in patterns:
                equal = np.array_equal(probe, p)
                if equal:
                    return min_probe, min_energy, iter
        #print(f'iter number: {iter} and energy: {energy}')
        iter += 1

    return min_probe, min_energy, iter
  
# Code from Dr. Schuman's notes on Hopfield networks  
def calculate_energy(probe, weights):
    energy = 0.0
    for i in range(num_neurons):
        for j in range(num_neurons):
            if(i != j):
                energy += weights[i,j]*probe[i]*probe[j]
    energy *= -0.5
    return energy


def rand_mnist_images(num_patterns):
    # Loop through dirs of mnist_png/testing/NUM_NAME
    # and get one random file to imprint.
    # Return the matrix of num_patterns x (28x28) to user
    patterns_matrix = []
    
    basedir = "./mnist_png/testing/"
    for i in range(num_patterns):
        # Add onto dir to get the directory for the specific number
        rand_num = np.random.randint(10)
        filesdir = basedir + str(rand_num) + "/"
        
        # Get the random image
        files = [os.path.join(filesdir, f) for f in os.listdir(filesdir) if os.path.isfile(os.path.join(filesdir, f))]

        # random index to get file
        ind = np.random.randint(len(files))
        num_img = files[ind]
        bin_img = png_to_binary_img(num_img)
        bin_arr = desquarify(bin_img)
        patterns_matrix.append(bin_arr)

        # plt.imshow(bin_img, cmap="binary")
        # plt.show()
    return np.array(patterns_matrix)

#MAIN
# Loop through increasing number of patterns (1->20)
def img_main():

    recall_count = np.zeros(m+1)
    recall_steps = []
    for i in range(1, m+1):
        
        
        #Make patterns (randomize or images)
        #patterns = np.random.choice([-1, 1], (i, num_neurons))

        patterns = rand_mnist_images(i)
        #Imprint those patterns and create weight matrix
        weights = imprintPatterns(patterns)
        
        recall_steps.append([])
        #Recall process, async vs sync or pick one
        for num, p in enumerate(patterns): # testing j number of probes with given weight matrix
            #noisy = p.copy() #Noisy probe
            #noisy_ind = np.random.choice(num_neurons, (int)(num_neurons * 0.3), replace=False)
            # could this also work?
            # noisy[noisy_ind] *= -1
            # for ind in noisy_ind:
            #     noisy[ind] *= -1
            # plt.imshow(squarify(p), cmap="binary")
            # plt.show()
            noisy = p.copy()
            noisy_ind = np.array(np.random.choice(num_neurons, (int)(num_neurons * 0.00), replace=False))
            #print(type(noisy_ind[0]))
            #print(type(noisy[0]))
            #print(noisy)   
            noisy[noisy_ind] *= -1
            probe = noisy
            # probe = noisy
            
            # plt.imshow(squarify(probe), cmap="binary")
            # plt.show()
            new_probe, min_e, iter = async_recall(probe, weights, patterns)
            print(f'Imprinting {i} patterns allowed probe {num} to converge in {iter} iterations {"(reached iteration threshold)" if iter == iter_threshold else ""}')
            
            # plt.imshow(squarify(np.array(new_probe)), cmap="binary")
            # plt.show()
            if(iter < iter_threshold):
                recall_count[i] += 1
        recall_count[i] /= i
        recall_steps[i - 1].append(iter)
        
    print(recall_count)

img_main()
"""
Things to look into
 - async vs sync recalling - thurs
 - Images rather than random
 - Noise?
 - Imprinted patterns
"""