import collections
from os import path
from typing import List, Dict, Any
import copy
import json
import matplotlib
from matplotlib.pyplot import axline, plot
import torch
import torch.nn as nn
import torchvision
import tqdm
import torchvision.datasets
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
from lenet import Lenet300100
from main_mnist import VALIDATION_ITERATIONS, PM_LIST, PM_LIST_REINIT, PRUNE_RATE

# Remove later
VALIDATION_ITERATIONS = np.arange(200, 50000, 200, dtype=int)
PM_LIST = [0, 3, 7, 12, 15, 18]
PM_LIST_REINIT = [0, 3, 7, 12, 15, 18]
PRUNE_RATE = 0.2

def plot_dict(plot_data: Dict[str, Any],
              colors: List[str],
              validation_iterations: np.ndarray,
              ax: matplotlib.axes.Axes):
    for label in plot_data.keys():
        for color in colors:
            if plot_data[label]['color'] == color:
                average = np.mean(plot_data[label]['accuracy'], axis=0)
                errors = np.vstack((np.amax(plot_data[label]['accuracy'], axis=0)-average, -np.amin(
                    plot_data[label]['accuracy'], axis=0)+average))
                ax.errorbar(validation_iterations, average,
                            yerr=errors, label=label, color=color)


def figure1(PM_LIST, VALIDATION_ITERATIONS, accuracies_array, losses_array):
    prune_rates = (1 - PRUNE_RATE) ** np.arange(0, max(PM_LIST)+1)*100
    es_index = np.argmin(np.mean(losses_array, 1), axis=1)
    average_es = VALIDATION_ITERATIONS[es_index]
    errors_es = np.vstack((VALIDATION_ITERATIONS[np.min(np.argmin(losses_array, axis=2), axis=1)] -
                          average_es, average_es-VALIDATION_ITERATIONS[np.max(np.argmin(losses_array, axis=2), axis=1)]))
    accuracies_es = np.take(np.mean(accuracies_array, 1), es_index)
    errors_accuracy_es = np.vstack((np.max(np.take(accuracies_array, np.argmin(losses_array, axis=2)), 1) -
                                    average_es, average_es-np.min(np.take(accuracies_array, np.argmin(losses_array, axis=2)), 1)))
    return prune_rates, average_es, errors_es, accuracies_es, errors_accuracy_es


# files = ["accuracies.npy", "accuracies_reinit.npy", "accuracies_random.npy",
#          "losses.npy", "losses_reinit.npy", "losses_random.npy"]
# accuracies_array, accuracies_array_reinit, accuracies_array_random, losses_array, losses_array_reinit, losses_array_random = [
#     np.load(file) for file in files]
files = ["accuracies.npy", "losses.npy"]
accuracies_array, losses_array = [np.load(file) for file in files]

prune_rates, average_es, errors_es, accuracies_es, errors_accuracy_es = figure1(
    PM_LIST, VALIDATION_ITERATIONS, accuracies_array, losses_array)
# prune_rates_random, average_es_random, errors_es_random, accuracies_es_random, errors_accuracy_es_random = figure1(
#     PM_LIST, VALIDATION_ITERATIONS, accuracies_array_random, losses_array_random)


fig, ax = plt.subplots(nrows=4, ncols=5, sharex=False, sharey=False)
i = 0
for row in ax:
    for col in row:
        col.plot(np.arange(200, 50000, 200), np.mean(losses_array, 1)[i])
        col.axvline(average_es[i], c='r')
        col.set_xlim(0, 15000)
        col.title.set_text(round(prune_rates[i], 1))
        i += 1
        if i == 19:
            break
    else:
        continue
    break
plt.show()


fig, ((ax1, ax2)) = plt.subplots(
    nrows=1, ncols=2, sharex=False, sharey=False)
ax1.errorbar(prune_rates, average_es, yerr=errors_es,
             color='red', label='Lenet')
# ax1.errorbar(prune_rates_random, average_es_random, yerr=errors_es_random, color='red', label ='random', ls='--')
ax1.set_xscale('log')
ax1.set_xlim(130, 0.1)
ax1.set_ylim(0, np.max(VALIDATION_ITERATIONS))
ax1.set_xlabel("Percent of weights remaining")
ax1.set_ylabel("Early-Stop Iteration (Val.)")
ax1.set_xticks([100, 41.1, 16.9, 7.0, 2.9, 1.2, 0.5, 0.2])
ax1.grid()

ax2.errorbar(prune_rates, accuracies_es,
             yerr=errors_accuracy_es, color='red', label='Lenet')
# ax2.errorbar(prune_rates_random, average_es_accuracy_random, yerr=errors_accuracy_es_random, color='red', label ='random', ls='--')
ax2.set_xscale('log')
ax2.set_xlim(130, 0.1)
ax2.set_ylim(0.9, 1)
ax2.set_xlabel("Percent of weights remaining")
ax2.set_ylabel("Accuracy at Early-Stop (Test)")
ax2.set_xticks([100, 41.1, 16.9, 7.0, 2.9, 1.2, 0.5, 0.2])
ax2.grid()
plt.show()


#Figure 3
plot_data = {}
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
for pm, color in zip(PM_LIST, colors):
    key = f"{(1 - PRUNE_RATE) ** pm * 100:.1f}"
    plot_data[key] = {}
    plot_data[key]["accuracy"] = accuracies_array[pm]
    plot_data[key]['color'] = color

# colors_reinit = ['pink', 'cyan']
# for pm, color in zip(PM_LIST_REINIT[1:], colors_reinit):
#     key = f"{(1 - PRUNE_RATE) ** pm * 100:.1f} (reinit)"
#     plot_data[key] = {}
#     plot_data[key]["accuracy"] = accuracies_array_reinit[pm]
#     plot_data[key]['color'] = color

fig, ((ax1, ax2, ax3)) = plt.subplots(
    nrows=1, ncols=3, sharex=False, sharey=False)
plot_dict(plot_data, ['green', 'orange', 'blue'],
            VALIDATION_ITERATIONS, ax1)
plot_dict(plot_data, ['green', 'red', 'blue',
            'purple', 'brown'], VALIDATION_ITERATIONS, ax2)
plot_dict(plot_data, ['green', 'orange', 'blue',
            'pink', 'cyan'], VALIDATION_ITERATIONS, ax3)

lines = []
labels = []
for ax in fig.axes:
    ax.set_xlim(0, 17500)
    ax.set_ylim(0.94, 0.99)
    ax.set_xticks([0, 5000, 10000, 15000])
    ax.grid()
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
legend = {label: line for label, line in zip(labels, lines)}
fig.legend(list(legend.values()), list(legend.keys()),
            loc='upper center', frameon=False, ncol=len(labels))
plt.show()
