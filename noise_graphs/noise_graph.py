import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('noise.csv')

#Line graph of averages as more patterns are imprinted
recall_ratio = df.groupby("noise_lvl").mean().iloc[:, 2:]
recall_ratio.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(0, 10):
    plt.plot(recall_ratio.iloc[i])

plt.title("Noisy Patterns Recallability")
plt.xlabel("Number of patterns imprinted")
plt.ylabel("Fraction of recalled patterns")
plt.show()

#Mean and standard deviation plotted
noises = np.arange(0.1, 1.1, 0.1)
for i in range(0, 10):
    sub_df = df[df["noise_lvl"] == noises[i]]
    means = df.groupby("noise_lvl").mean().iloc[i, 2:]
    stds = df.groupby("noise_lvl").std().iloc[i, 2:]
    means.index = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    stds.index = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    plt.plot(means.index, means.values)
    plt.fill_between(means.index, means - stds, means + stds, alpha=0.3)

    plt.title("Noisy Patterns Recallability")
    plt.xlabel("Number of patterns imprinted")
    plt.ylabel("Fraction of recalled patterns")
    plt.savefig(os.path.join('noise_graphs', str(round(noises[i], 1)) + "_Mean_STDs.png"))
    plt.clf()