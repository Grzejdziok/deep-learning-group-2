import string
from typing import List
import matplotlib
from matplotlib.pyplot import plot
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets
import torch.nn.utils.prune as prune
import numpy as np
import copy
import matplotlib.pyplot as plt
from lenet import Lenet300100


def plot_dict(plot_data: dict[string, dict],
              colors:List[str],
              validation_iterations:np.ndarray,
              ax: matplotlib.axes.Axes):
    for label in plot_data.keys():
        for color in colors:
            if plot_data[label]['color']==color:
                average = np.mean(plot_data[label]['accuracy'], axis=0)
                errors = np.vstack((np.amax(plot_data[label]['accuracy'], axis = 0)-average, -np.amin(plot_data[label]['accuracy'], axis = 0)+average))
                ax.errorbar(validation_iterations, average, yerr = errors, label=label, color=color)


def train_model(train_data_loader: torch.utils.data.DataLoader,
                test_data_loader: torch.utils.data.DataLoader,
                USE_CUDA: bool,
                model: nn.Module,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                validate_at: List[int],
                ) -> np.ndarray:
    iterations_so_far = 0
    iter_train_data_loader = iter(train_data_loader)
    total_iterations = max(validate_at)
    accuracies = []
    while iterations_so_far < total_iterations:
        try:
            batch_input, batch_target = next(iter_train_data_loader)
            if USE_CUDA:
                batch_input = batch_input.cuda()
                batch_target = batch_target.cuda()
        except StopIteration:
            iter_train_data_loader = iter(train_data_loader)
            continue
        model.train()
        batch_output_prob = model(batch_input)
        loss = criterion(batch_output_prob, batch_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        iterations_so_far += 1

        if iterations_so_far in validate_at:
            # Find accuracy
            model.eval()
            num_hits = 0.
            for test_batch_input, test_batch_target in test_data_loader:
                if USE_CUDA:
                    test_batch_input = test_batch_input.cuda()
                    test_batch_target = test_batch_target.cuda()
                with torch.no_grad():
                    test_batch_output_prob = model(test_batch_input)
                    test_batch_output = torch.argmax(
                        test_batch_output_prob, dim=1)
                    num_hits += (test_batch_output ==
                                 test_batch_target).sum().float().item()
            accuracy = num_hits / len(test_dataset)
            accuracies.append(accuracy)
    accuracies = np.asarray(accuracies)
    return accuracies


def main_fig3(USE_CUDA: bool,
              LEARNING_RATE: float,
              PRUNE_RATE: float,
              validation_iterations: np.ndarray,
              pm_list: List[int],
              random_init: bool,
              ):
    model = Lenet300100()
    if USE_CUDA:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    weights_original = []  # Save original weights at the beginning
    if isinstance(model, Lenet300100):  # specific to Lenet
        for i in [0, 2, 4]:
            weights_original.append(copy.deepcopy(model.layers[i].weight))
    else:
        raise NotImplementedError

    for pm in range(max(pm_list)+1):  # Iterate over pruning rates
        print(
            f"Reinit:{random_init}; Training at params ratio: {(1-PRUNE_RATE)**pm:.3f}, active parameters: {sum(model.layers[2*w].weight.count_nonzero().item() for w in range (3))}")
        accuracies = train_model(train_data_loader, test_data_loader,
                                 USE_CUDA, model, criterion, optimizer, validation_iterations)
        # Prune
        if isinstance(model, Lenet300100):  # specific to Lenet
            prune.random_unstructured(
                model.layers[0], name="weight", amount=PRUNE_RATE)
            prune.random_unstructured(
                model.layers[2], name="weight", amount=PRUNE_RATE)
            # Last layer at half the rate - page 22
            prune.random_unstructured(
                model.layers[4], name="weight", amount=PRUNE_RATE/2)
        else:
            raise NotImplementedError
        # Update
        mask = list(model.named_buffers())
        if not random_init:
            if isinstance(model, Lenet300100):  # specific to Lenet
                for i in range(3):
                    weights_original[i] = weights_original[i]*mask[i][1]
                    model.layers[2*i].weight = weights_original[i]
                if pm in pm_list:
                    append_accuracies(
                        plot_data, f"{(1-PRUNE_RATE)**pm*100:.1f}", accuracies)
            else:
                raise NotImplementedError
        else:
            if isinstance(model, Lenet300100):  # specific to Lenet
                for i in range(3):
                    weights_fake = torch.empty(
                        model.layers[2*i].weight.shape, device='cuda').normal_(mean=MU, std=STD)
                    weights_fake = weights_fake*mask[i][1]
                    model.layers[2*i].weight = weights_fake
                if pm in pm_list:
                    append_accuracies(
                        plot_data, f"{(1-PRUNE_RATE)**pm*100:.1f}"+" (reinit)", accuracies)
                # plot_data[f"{(1-PRUNE_RATE)**pm*100:.1f}"+" (reinit)"]["accuracy"].append([accuracies])
            else:
                raise NotImplementedError


def append_accuracies(dict_obj, key, value):
    if key in dict_obj:
        try:
            dict_obj[key]['accuracy'] = np.vstack((dict_obj[key]['accuracy'], value))
        except:
            raise "accuracy subkey not in dictionary"
    else:
        dict_obj[key] = {}
        dict_obj[key]['accuracy'] = value


if __name__ == "__main__":

    BATCH_SIZE = 60
    LEARNING_RATE = 1.2e-3
    validation_iterations = np.arange(100, 50000, 100, dtype=int)
    # P_m's from figure 3 - these are the exponents of 0.8 to get to roughly the Pm's for figure 3
    pm_list = [0, 3, 7, 12, 15, 18]
    # plot_data = np.zeros((len(pm_list)+2, len(validation_iterations))) #+2 for the reinit lines
    plot_data = {}
    PRUNE_RATE = 0.2
    USE_CUDA = torch.cuda.is_available()
    # seemingly the normalization parameters according to LeCun (1998)
    MU = 0.1/1.275
    STD = 1/1.275

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MU, std=STD),
        torchvision.transforms.Lambda(lambda tensor: tensor.reshape(-1)),
    ])
    train_dataset = torchvision.datasets.MNIST(
        root="./.mnist", transform=transform, download=True, train=True)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(
        root="./.mnist", transform=transform, download=True, train=False)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for i in range(5):
        print(f"---ITERATION: {i+1}---")
        main_fig3(USE_CUDA, LEARNING_RATE, PRUNE_RATE,
                validation_iterations, pm_list, random_init=False)
        main_fig3(USE_CUDA, LEARNING_RATE, PRUNE_RATE,
                validation_iterations, pm_list=[3, 7], random_init=True)

    # Plot figure 3
    colours = ['blue', 'orange', 'green', 'red',
               'purple', 'brown', 'pink', 'cyan']
    for i, key in enumerate(plot_data):
        plot_data[key]['color'] = colours[i]


    fig, ((ax1, ax2, ax3)) = plt.subplots(
        nrows=1, ncols=3, sharex=False, sharey=False)
    plot_dict(plot_data, ['green', 'orange', 'blue'], validation_iterations, ax1)
    plot_dict(plot_data, ['green', 'red', 'blue', 'purple', 'brown'], validation_iterations, ax2)
    plot_dict(plot_data, ['green', 'orange', 'blue', 'pink', 'cyan'], validation_iterations, ax3)

    lines = []
    labels = []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    legend = {label: line for label, line in zip(labels, lines)}
    fig.legend(list(legend.values()), list(legend.keys()),
               loc='upper center', frameon=False, ncol=len(labels))
    plt.show()
