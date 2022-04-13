from typing import List, Dict, Any
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


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
                            yerr=errors, label=label, color=color, errorevery=10, capsize=2)


def figure1(PM_LIST, VALIDATION_ITERATIONS, accuracies_array, losses_array, prune_rate):
    prune_rates = (1 - prune_rate) ** np.arange(0, max(PM_LIST)+1)*100
    es_index = np.argmin(losses_array, 2)
    average_es = np.mean(VALIDATION_ITERATIONS[es_index], axis=1)
    errors_es = np.vstack((VALIDATION_ITERATIONS[np.min(es_index, axis=1)] -
                          average_es, average_es-VALIDATION_ITERATIONS[np.max(es_index, axis=1)]))
    accuracies_es = np.squeeze(np.take_along_axis(accuracies_array, np.expand_dims(es_index, 2), axis=2), axis=2)
    mean_accuracies_es = np.mean(accuracies_es, 1)
    max_accuracies_es = np.max(accuracies_es, 1)
    min_accuracies_es = np.min(accuracies_es, 1)
    errors_accuracy_es = np.vstack((max_accuracies_es - mean_accuracies_es, mean_accuracies_es-min_accuracies_es))
    return prune_rates, average_es, errors_es, mean_accuracies_es, errors_accuracy_es


if __name__ == "__main__":
    # Load data
    f = open('data.json')
    data = json.load(f)
    PM_LIST = data['PM_LIST']
    VALIDATION_ITERATIONS = np.array(data['VALIDATION_ITERATIONS'])
    PRUNE_RATE = data['PRUNE_RATE'] if isinstance(data['PRUNE_RATE'], float) else data['PRUNE_RATE'][0]
    accuracies_array = np.array(data['test_accuracies'])
    losses_array = np.array(data['val_losses'])

    f = open('data_random.json')
    data = json.load(f)
    PM_LIST_random = data['PM_LIST']
    VALIDATION_ITERATIONS_random = np.array(data['VALIDATION_ITERATIONS'])
    PRUNE_RATE_random = data['PRUNE_RATE'] if isinstance(data['PRUNE_RATE'], float) else data['PRUNE_RATE'][0]
    accuracies_array_random = np.array(data['test_accuracies'])
    losses_array_random = np.array(data['val_losses'])

    f = open('data_reinit.json')
    data = json.load(f)
    PM_LIST_reinit = data['PM_LIST']
    VALIDATION_ITERATIONS_reinit = np.array(data['VALIDATION_ITERATIONS'])
    PRUNE_RATE_reinit = data['PRUNE_RATE'] if isinstance(data['PRUNE_RATE'], float) else data['PRUNE_RATE'][0]
    accuracies_array_reinit = np.array(data['test_accuracies'])
    losses_array_reinit = np.array(data['val_losses'])


    prune_rates, average_es, errors_es, accuracies_es, errors_accuracy_es = figure1(
        PM_LIST, VALIDATION_ITERATIONS, accuracies_array, losses_array, PRUNE_RATE)
    prune_rates_random, average_es_random, errors_es_random, accuracies_es_random, errors_accuracy_es_random = figure1(
        PM_LIST_random, VALIDATION_ITERATIONS_random, accuracies_array_random, losses_array_random, PRUNE_RATE_random)

    fig, ((ax1, ax2)) = plt.subplots(
        nrows=1, ncols=2, sharex=False, sharey=False)
    ax1.errorbar(prune_rates, average_es, yerr=errors_es,
                color='red', label='Lenet')
    ax1.errorbar(prune_rates_random, average_es_random,
                yerr=errors_es_random, color='red', label='random', ls='--')
    ax1.set_xscale('log')
    ax1.set_xlim(130, 0.1)
    ax1.set_ylim(0, np.max(VALIDATION_ITERATIONS))
    ax1.set_xlabel("Percent of weights remaining")
    ax1.set_ylabel("Early-Stop Iteration (Val.)")
    ax1.set_xticks([100, 41.1, 16.9, 7.0, 2.9, 1.2, 0.5, 0.2])
    ax1.set_xticklabels([100, 41.1, 16.9, 7.0, 2.9, 1.2, 0.5, 0.2])
    ax1.minorticks_off()
    ax1.grid()

    ax2.errorbar(prune_rates, accuracies_es,
                yerr=errors_accuracy_es, color='red', label='Lenet')
    ax2.errorbar(prune_rates_random, accuracies_es_random,
                yerr=errors_accuracy_es_random, color='red', label='random', ls='--')
    ax2.set_xscale('log')
    ax2.set_xlim(130, 0.1)
    ax2.set_ylim(0.85, 1)
    ax2.set_xlabel("Percent of weights remaining")
    ax2.set_ylabel("Accuracy at Early-Stop (Test)")
    ax2.set_xticks([100, 41.1, 16.9, 7.0, 2.9, 1.2, 0.5, 0.2])
    ax2.set_xticklabels([100, 41.1, 16.9, 7.0, 2.9, 1.2, 0.5, 0.2])
    ax2.minorticks_off()
    ax2.grid()
    plt.show()


    # Figure 3
    plot_data = {}
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for pm, color in zip(PM_LIST, colors):
        key = f"{(1 - PRUNE_RATE) ** pm * 100:.1f}"
        plot_data[key] = {}
        plot_data[key]["accuracy"] = accuracies_array[pm]
        plot_data[key]['color'] = color

    colors_reinit = ['pink', 'cyan']
    for pm, color in zip(PM_LIST_reinit[1:], colors_reinit):
        key = f"{(1 - PRUNE_RATE) ** pm * 100:.1f} (reinit)"
        plot_data[key] = {}
        plot_data[key]["accuracy"] = accuracies_array_reinit[pm]
        plot_data[key]['color'] = color

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
        ax.set_xlabel("Training Iterations")
        ax.set_ylim(0.94, 0.99)
        ax.set_ylabel("Test Accuracy")
        ax.set_xticks([0, 5000, 10000, 15000])
        ax.grid()
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    legend = {label: line for label, line in zip(labels, lines)}
    fig.legend(list(legend.values()), list(legend.keys()),
            loc='upper center', frameon=False, ncol=len(labels))
    plt.show()
