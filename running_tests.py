import numpy as np
import pandas as pd

#Create dataframe
columns = ['experiment', 'iter', 'noise_lvl', 'num_neurons_updated', 'patterns_imprinted(m)', 'recall_ratio', 'average_hamming', 'average_steps']
df = pd.DataFrame(columns=columns)

#Hyper-parameters
#Number of patterns is technically a hyper parameter but that is done in the main code
noise = np.arange(0, 0.6, 0.1) #6
num_neurons_updated = np.arange(0.1, 1.1, 0.1) #10

#6 * 10 = 60 experiments
#60 experiments * 10 iterations = 600 tests (600 * 20 patters = 12,000)
for i in noise:
    for j in num_neurons_updated:

        #Create 20 patterns

        for iter in range(0, 10):
            ret_df = 1 #Call levi's code using this patterns created above

            new_row = [i, j, ret_df] 
            df = pd.concat([df, pd.Series(new_row, index = df.columns).to_frame().T], axis=0, ignore_index=True)

df.to_csv('async_data.csv')