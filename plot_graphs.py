import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

root = "saved_models/"
algorithms = ["DQN", "duellingDQN", "doubleDQN", "duelling_doubleDQN"]
file_paths = [root + a + "_stats.log" for a in algorithms]

labels = []
rewards = []
losses = []

for f in file_paths:
    if os.path.exists(f):
        df = pd.read_csv(f, header=None)
        labels.append(df.loc[0][0])

        reward = np.asarray(df.iloc[0, 2:].values)
        rewards.append(reward)

        loss = np.asarray(df.iloc[1, 2:].values)
        losses.append(loss)

def plot_graph(y_vals, x_label, y_label, f_name):
    for i in range(len(y_vals)):
        Y = y_vals[i][::10]
        X = list(range(len(Y)))
        plt.plot(X, Y, label=labels[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f_name)
    plt.show()

plot_graph(rewards, "Number of Episodes", "Reward obtained", "images/reward_comparision.png")
plot_graph(losses, "Number of Episodes", "Training loss", "images/loss_comparision.png")
