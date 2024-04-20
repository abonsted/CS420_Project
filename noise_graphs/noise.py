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

def async_recall(probe, weights, p, patterns):
    start = probe.copy()
    vis = []
    for steps in range(50): #Number of steps (sync)
        #Randomize order
        order = np.array(range(num_neurons))
        np.random.shuffle(order)
        order = order[0:25]
        vis.append(probe.copy())

        #Update neurons
        for i in order:
            h = 0
            for j in range(num_neurons):
                h += weights[i, j] * probe[j]
            probe[i] = sign(h)

        #Check to see if pattern was recalled
        if(np.array_equal(probe, p)):
            print("SAME")
            return steps, probe, vis 

    print("FINAL")
    print(probe)
    return steps, probe, vis

# def calculate_energy(probe, weights):
#     energy = 0.0
#     for i in range(num_neurons):
#         for j in range(num_neurons):
#             if(i != j):
#                 energy += weights[i,j]*probe[i]*probe[j]
#     energy *= -0.5
#     return energy

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
            print("INTIAL")
            print(p)
            noisy = p.copy()
            noisy_ind = np.random.choice(num_neurons, (int)(num_neurons * noise_lvl), replace=False)
            for ind in noisy_ind:
                noisy[ind] *= -1
            probe = noisy
            if(np.array_equal(probe, p)):
                print("SAME")
            iter, new_probe, vis = async_recall(probe, weights, p, patterns)
            plt.imshow(vis, cmap="binary")
            plt.show()
            if(num == 2):
                exit()

    return recall_count[1:-1]

#MAIN here
# columns = ['noise_lvl', 'iter'] + [f'{i}_patterns' for i in range(1, m)]
# df = pd.DataFrame(columns=columns)

# noise_lvls = np.arange(0.1, 1.1, 0.1)
# print(noise_lvls)
# for i in noise_lvls:
#     print("Starting ", i)
#     for iter in range(1, 21):
#         print("Iter ", iter)
#         recall_ratio = hop_net(i)
#         new_row = np.concatenate(([i, iter], recall_ratio), axis=0) 
#         df = pd.concat([df, pd.Series(new_row, index = df.columns).to_frame().T], axis=0, ignore_index=True)
# print(df)

# df.to_csv('noise.csv')

hop_net(0.49)