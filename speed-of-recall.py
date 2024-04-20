# Modifiations of main.py written by Andrew Bonsted
# In preparations for the week of 4/15/24
# Measuring the speed of recall (number of asynchronous steps needed vs. number of patterns imprinted)\
# 
# NOTE: Untar the mnist_png_testing.tar.gz to be able to use the functions in this file
#       properly

from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
from io import BytesIO
from PIL import Image as Image

# make a seed for more consistent results
# seed = 1794
# np.random.seed(seed)

#Global vals
m = 20                  # "m" fundamental memories to imprint
num_neurons = 22 * 22   # trimmed size of the images from the original 28x28 MNIST images
iter_threshold = 50     # upper bound of time step iterations to perform
tolerance = 95.0        # % amount of the image is recalled correctly


# convert png image from PIL to usable format
def png_to_hop_img(infile_path):
    # img = Image.open(png_path)
    
    # grayscale_img = img.convert("L")
    
    # bin_threshold = 128
    # binary_img = grayscale_img.point(lambda x: -1 if x < bin_threshold else 1)
    # print(grayscale_img)
    # bin_arr = np.array(binary_img)
    
    # Open the image as a grey scale image and convert the default uint8 data type to something that can be 
    # turned negative
    img_arr = np.array(Image.open(infile_path).convert("L")).astype('int32')
    # Make the binary image
    all_cols = img_arr[:, :]
    all_cols[all_cols == 0] = -1
    all_cols[all_cols > 0] = 1
    
    # The manual square trimming occurs here: 
    # Go from 28x28 to 22x22 by removing 3 columns and 3 rows from the original image
    # on all sides
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
def async_recall(probe, weights, original_pattern):
    
    iter  = 0
    energy_after = 0
    energy_prior = calculate_energy(probe, weights)
    for time_step in range(iter_threshold):
        order = np.array(range(num_neurons))
        np.random.shuffle(order)
        
        #print(order)
        # for o in order:
        #     h = 0
        #     for neighbor in weights[o]:
        #         h += neighbor * probe[o]
        #     probe[o] = sign(h)
        # order = np.random.choice(num_neurons, 200, replace=False)
        # np.random.shuffle(order)
        
        for i in order:
            h = 0
            
            for j in range(weights.shape[0]):
                h += weights[i,j]*probe[j]
            probe[i] = sign(h)
            
        # calculate energy here
        energy_after = calculate_energy(probe, weights)
        
        #print(f'energy before: {energy_prior}  energy after: {energy_after}')
        if energy_prior == energy_after:
            return probe, energy_after, time_step + 1
        
        energy_prior = energy_after
        iter += 1
        
    return probe, energy_after, iter

    # This is what we originally had:
    # Only exiting when the iterations were through or there was an exact match
    # between the probe and one of the patterns imprinted
    # while iter < iter_threshold:
        
    #     neuron_index = np.random.randint(0, len(probe))
    #     before = probe[neuron_index]
    #     after = sign(np.dot(weights[neuron_index], probe))
    #     probe[neuron_index] = after
        
    #     # Recompute energy if a state changed
    #     #Check energy threshold
    #     energy = calculate_energy(probe, weights)
    #     if(energy < min_energy):
    #         print("NEW MIN: ", energy, " on iteration: ", iter)
    #         min_energy = energy
    #         min_probe = probe

    #         for p in patterns:
    #             equal = np.array_equal(probe, p)
    #             if equal:
    #                 return min_probe, min_energy, iter
    #     #print(f'iter number: {iter} and energy: {energy}')
    #     iter += 1

  
# Code from Dr. Schuman's notes on Hopfield networks  
def calculate_energy(probe, weights):
    energy = 0.0
    for i in range(num_neurons):
        for j in range(num_neurons):
            if(i != j):
                energy += weights[i,j]*probe[i]*probe[j]
    energy *= -0.5
    return energy

# Returns a 2D array of size (num_patterns, num_neurons) of random
# MNIST images (that are trimmed from 28x28 to 22x22 by internal helper
# function)
def rand_mnist_matrix(num_patterns):
    # Loop through dirs of mnist_png/testing/NUM_NAME
    # and get one random file to imprint.
    # Return the matrix of num_patterns x (28x28) to user
    patterns_matrix = []
    chosen_str = ""
    basedir = "./mnist_png/testing/"
    for i in range(num_patterns):
        # Add onto dir to get the directory for the specific number
        rand_num = np.random.randint(10)
        chosen_str += f'{rand_num}, '
        filesdir = basedir + str(rand_num) + "/"
        
        # Get the random image
        files = [os.path.join(filesdir, f) for f in os.listdir(filesdir) if os.path.isfile(os.path.join(filesdir, f))]

        # random index to get file
        ind = np.random.randint(len(files))
        num_img = files[ind]
        hop_img = png_to_hop_img(num_img)
        hop_arr = desquarify(hop_img)
       
        patterns_matrix.append(hop_arr)

        # plt.imshow(bin_img, cmap="binary")
        # plt.show()
    # Display what random patterns were chosen to imprint into the weight matrix 
    print("NUMBERS CHOSEN: ", chosen_str)
    return np.array(patterns_matrix)

# Just returns a single random 1D probe that can be used in isolation or appended to another list
def rand_mnist_pattern(tracking_arr):
    basedir = "./mnist_png/testing/"

    rand_num = np.random.randint(10)
    tracking_arr.append(rand_num)
    filesdir = basedir + str(rand_num) + "/"
    
    # Get the random image
    files = [os.path.join(filesdir, f) for f in os.listdir(filesdir) if os.path.isfile(os.path.join(filesdir, f))]
    # random index to get file
    ind = np.random.randint(len(files))
    num_img = files[ind]
    hop_img = png_to_hop_img(num_img)
    hop_arr = desquarify(hop_img)
    
    return hop_arr
        
    
# From Eli's code (not hamming distance but similar idea)
# Returns percentage recovered as a percentage value [0, 100]
def resemblance(image, probe):
    matching = np.sum(image == probe)
    total = image.shape[0]
    res = (matching / total) * 100
    return res

# Returns the number of positions that different from a given image
# and the resulting recalled image
# Original and recalled are assumed to be 1D np arrays
def hamming_distance(original, recalled):
    return np.sum(original != recalled)
#MAIN
# Loop through increasing number of patterns (1->20)
def img_main():

    recall_count = np.zeros(m+1)
    recall_steps = []
    patterns = []
    tracking_arr = []  # keep track of the corresponding number for the i'th pattern
    for i in range(1, m+1):
               
        #Make patterns (randomize or images)
        #patterns = np.random.choice([-1, 1], (i, num_neurons))

        # Generate i random mnist_images
        # THIS ASSUMES THE mnist_png_testing.tar.gz FILE WAS UNTARRED IN CURRENT DIRECTORY
        # This also generates i entirely new patterns. something like the following should probably be used
        # patterns.append(rand_mnist_patterns)
        #patterns = rand_mnist_matrix(i)
        new_pattern = rand_mnist_pattern(tracking_arr)
        patterns.append(new_pattern)
        print(f'The current imprinted patterns: {tracking_arr}')
        #Imprint those patterns and create weight matrix
        weights = imprintPatterns(patterns)
        
        # Initial recall_steps for the current iterations
        # recall_steps[i] will append i values to it, representing the iterations it 
        # took to recall the recall_steps[i][j]'th pattern
        recall_steps.append([])
        
        #Recall process, async vs sync or pick one
        for num, p in enumerate(patterns): # testing j number of probes with given weight matrix
            
            # Show the original image before adding noise
            # plt.imshow(squarify(p), cmap="binary")
            # plt.show()
            
            # Create the noisy vector to use as the initial probe
            noisy = p.copy()
            noise_amount = 0.1
            # Pseudorandomly select (num_neurons * noise_amount) unique indices from the original image
            # to flip to the other state
            noisy_ind = np.array(np.random.choice(num_neurons, (int)(num_neurons*noise_amount), replace=False))
            noisy[noisy_ind] *= -1
            
            # Show the noisy image
            # plt.imshow(squarify(noisy), cmap="binary")
            # plt.show()
            
            # Perform the recall with the established noisy vector and the imprinted weights matrix
            # the 'p' parameter currently isn't used, but it's here for coherenecy with other versions
            # of the async_recall function betwen us
            new_probe, min_e, iter = async_recall(noisy, weights, p)
            ham_dist = hamming_distance(p, new_probe)
            resemb = resemblance(p, new_probe)
            
            print(f'Imprinting {i} patterns allowed probe {num} to converge in {iter} iterations {"(reached iteration threshold)" if iter == iter_threshold else ""}')
            print(f'The resemblance was: {resemb}')
            print(f'The Hamming distance between the original and the resulting image are: {ham_dist}')
            
            # Show the recalled image
            # plt.imshow(squarify(np.array(new_probe)), cmap="binary")
            # plt.show()
            
            # Only count a image to be recalled if it converged in less than the iteration threshold
            # and the resemble is close enough to the tolerance established
            if(iter < iter_threshold and resemb >= tolerance):
                recall_count[i] += 1
                
        # Avergage the number recalled successfully with tolerance and the steps required amongst the patterns 
        # for that recall to occur       
        recall_count[i] /= i
        recall_steps[i - 1].append(iter)
        
    print("Recall count: ", recall_count)


if __name__ == '__main__':
    # May want to repeat the results of image main multiple times and average results
    # with different random images generated at each stage
    img_main()

"""
Things to look into
 - async vs sync recalling - thurs
 - Images rather than random
 - Noise?
 - Imprinted patterns
"""

"""
Hyperparamters to test
 - Number of neurons to update in async
 - Noise values upong imprinting 
       Could use seed to ensure the same sequences of patterns
        are imprinted with different noise values so  the specific
        images imprinted does not change with the noise level
 - Async. vs synchronous results?
 - Resemblance / Hamming distance performance vs. Patterns imprinted
"""

"""
Number of imprinted patterns could affect:
 - Number of time steps needed for convergence?
 - Performance for recall?
 - Resiliance with respect to noise?
"""