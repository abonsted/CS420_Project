import SOR as sor
import numpy as np
import pandas as pd
import time

#Create dataframe
columns = ['experiment', 'iter', 'noise_lvl', 'num_neurons_updated', 'patterns_imprinted(m)', 'avg_recall_ratio', 'avg_hamming', 'avg_resemblance' ,'avg_steps']
async_df = pd.DataFrame(columns=columns)
sync_df = pd.DataFrame(columns=columns)

#Hyper-parameters
#Number of patterns is technically a hyper parameter but that is done in the main code
noise = np.arange(0.3, 0.6, 0.1) #6
num_neurons_updated = np.arange(0.1, 1.1, 0.1) #10

#6 * 10 = 60 experiments
#60 experiments * 10 iterations = 600 tests (600 * 20 patterns = 12,000)
ex_num = 0
for i in noise:
    for j in num_neurons_updated:
        print("EXPERIMENT ", ex_num)
        start = time.time()

        #Create 20 patterns
        patterns = sor.rand_mnist_matrix(20)

        #Async
        for iter in range(0, 10):
            print("ASYNC ITERATION ", iter)
            ret_df = sor.experiment_main(patterns, i, j, ex_num, iter)
            async_df = pd.concat([async_df, ret_df], axis=0, ignore_index=True)

        #Sync 
        for iter in range(0, 10):
            print("SYNC ITERATION ", iter)
            ret_df = sor.sync_experiment_main(patterns, i, j, ex_num, iter)
            sync_df = pd.concat([sync_df, ret_df], axis=0, ignore_index=True)
        ex_num += 1
        
        print(time.time() - start)


async_df.to_csv('async_data.csv')
sync_df.to_csv('sync_data.csv')