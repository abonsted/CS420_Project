import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import os

def round_vals(value):
    return round(value, 2)

def noise_graphing(df, amount):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df["avg_resemblance"], color='b')
    ax2.plot(df["avg_hamming"], color='r')

    plt.suptitle("Noise Level Effect on Network's Recallability")        
    plt.title(f"{amount} Neurons Updated Each Step")
    ax1.set_xlabel("Noise Level (%)")
    ax1.set_ylabel("Average Resemblance (%)")
    ax2.set_ylabel("Average Hamming Distance")
    ax2.set_ylim(0, 300)

    leg1 = mp.Patch(color='b', label='Average Resemblance')
    leg2 = mp.Patch(color='r', label='Average Hamming Dist')

    ax1.legend(handles=[leg1, leg2])

    plt.savefig(os.path.join('andrew-graphs', amount + "NoiseLevel_Recallability.png"))

async_df = pd.read_csv("async_data.csv")
async_df["num_neurons_updated"] = async_df["num_neurons_updated"].apply(round_vals)

# GRAPHS FOR A RANGE OF num_neurons_updated
    #Low = 0.1-0.3
    #Medium = 0.4-0.7
    #High = 0.8-1.0

# low_neurons_updated = async_df[async_df["num_neurons_updated"] < 0.4]
# mid_neurons_updated = async_df[(async_df["num_neurons_updated"] >= 0.4) & (async_df["num_neurons_updated"] <= 0.7)]
# high_neurons_updated = async_df[(async_df["num_neurons_updated"] > 0.7)]

# noise_lvls1 = low_neurons_updated.groupby("noise_lvl").mean()
# noise_lvls2 = mid_neurons_updated.groupby("noise_lvl").mean()
# noise_lvls3 = high_neurons_updated.groupby("noise_lvl").mean()

# noise_graphing(noise_lvls1, "Low")
# noise_graphing(noise_lvls2, "Medium")
# noise_graphing(noise_lvls3, "High")



#GRAPHS FOR EACH num_neurons_updated VALUES

# for i in np.arange(0.1, 1.1, 0.1):
#     i = round(i, 2)
#     print(i)
#     sub_df = async_df[async_df["num_neurons_updated"] == i]
#     sub = sub_df.groupby("noise_lvl").mean()

#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()
#     ax1.plot(sub["avg_resemblance"], color='b')
#     ax2.plot(sub["avg_hamming"], color='r')

#     plt.suptitle("Noise Level Effect on Network's Recallability")        
#     plt.title(f"{i * 100}% Neurons Updated Each Step")
#     ax1.set_xlabel("Noise Level (%)")
#     ax1.set_ylabel("Average Resemblance (%)")
#     ax2.set_ylabel("Average Hamming Distance")
#     ax2.set_ylim(0, 300)

#     leg1 = mp.Patch(color='b', label='Average Resemblance')
#     leg2 = mp.Patch(color='r', label='Average Hamming Dist')

#     ax1.legend(handles=[leg1, leg2])

#     plt.savefig(os.path.join('andrew-graphs', str(i) + "NoiseLevel_Recallability.png"))