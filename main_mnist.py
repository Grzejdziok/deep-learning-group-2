import collections
from typing import List, Dict, Any
import copy
import json
import matplotlib
from matplotlib.pyplot import plot
import torch
import torch.nn as nn
import torchvision
import tqdm
import torchvision.datasets
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
from lenet import Lenet300100


def plot_dict(plot_data: Dict[str, Any],
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
                validate_at: np.ndarray,
                ) -> np.ndarray:
    iterations_so_far = 0
    iter_train_data_loader = iter(train_data_loader)
    total_iterations = max(validate_at)
    accuracies = []
    progress_bar = tqdm.tqdm(total=total_iterations)
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
        progress_bar.update()

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
    progress_bar.close()
    accuracies = np.asarray(accuracies)
    return accuracies


def prune_model(model: nn.Module, prune_ratio_hidden: float, prune_ratio_output: float) -> nn.Module:
    if isinstance(model, Lenet300100):
        prune.l1_unstructured(model.layers[0], name="weight", amount=prune_ratio_hidden)
        prune.l1_unstructured(model.layers[2], name="weight", amount=prune_ratio_hidden)
        # Last layer at half the rate - page 22
        prune.l1_unstructured(model.layers[4], name="weight", amount=prune_ratio_output)
    else:
        raise NotImplementedError()
    return model


def load_original_weights(model: nn.Module, model_original: nn.Module) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            if prune.is_pruned(module):
                mask = module.weight_mask
                prune.remove(module, 'weight')
                module.weight.data = model_original.get_submodule(name).weight.data.clone()
                module.bias.data = model_original.get_submodule(name).bias.data.clone()
                prune.custom_from_mask(module, 'weight', mask)
            else:
                module.weight.data = model_original.get_submodule(name).weight.data.clone()
                module.bias.data = model_original.get_submodule(name).bias.data.clone()
    return model


def random_reinit(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            if prune.is_pruned(module):
                mask = module.weight_mask
                prune.remove(module, 'weight')
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
                prune.custom_from_mask(module, 'weight', mask)
            else:
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
    return model


def run_iterative_pruning(
        num_executions: int,
        num_prunings: int,
        validation_iterations: np.ndarray,
        train_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader,
        prune_rate: float,
        random_init: bool,
) -> np.ndarray:
    accuracies_array = np.zeros((num_prunings+1, num_executions, validation_iterations.shape[0]))
    for i in range(num_executions):
        print(f"---ITERATION: {i + 1}, RANDOM_INIT={random_init}---")
        model = Lenet300100()
        model = random_reinit(model)
        if USE_CUDA:
            model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

        model_original = copy.deepcopy(model)  # Save original weights at the beginning
        for pm in range(num_prunings+1):
            print(f"Training at params ratio: {(1 - PRUNE_RATE) ** pm:.3f}, "
                  f"active parameters: {sum(model.layers[2 * w].weight.count_nonzero().item() for w in range(3))}")
            accuracies = train_model(train_data_loader, test_data_loader, USE_CUDA, model, criterion, optimizer,
                                     validation_iterations)
            model = prune_model(model=model, prune_ratio_hidden=prune_rate, prune_ratio_output=prune_rate / 2)
            if not random_init:
                model = load_original_weights(model=model, model_original=model_original)
            else:
                model = random_reinit(model)
            accuracies_array[pm, i, :] = accuracies
            np.save(f"data{'_random_init' if random_init else ''}.npy", accuracies_array)
    return accuracies_array


if __name__ == "__main__":

    BATCH_SIZE = 60
    LEARNING_RATE = 1.2e-3
    VALIDATION_ITERATIONS = np.arange(200, 50001, 200, dtype=int)
    # P_m's from figure 3 - these are the exponents of 0.8 to get to roughly the Pm's for figure 3
    PM_LIST = [0, 3, 7, 12, 15, 18]
    PM_LIST_REINIT = [0, 3, 7]
    NUM_PRUNINGS = max(PM_LIST)
    NUM_PRUNINGS_REINIT = max(PM_LIST_REINIT)
    # plot_data = np.zeros((len(pm_list)+2, len(validation_iterations))) #+2 for the reinit lines
    NUM_EXECUTIONS = 5
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
    train_dataset, val_dataset = torch.utils.data.random_split(torchvision.datasets.MNIST(
        root="./.mnist", transform=transform, download=True, train=True), (55000, 5000))
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = torchvision.datasets.MNIST(
        root="./.mnist", transform=transform, download=True, train=False)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    accuracies_array = run_iterative_pruning(num_executions=NUM_EXECUTIONS,
                                             num_prunings=NUM_PRUNINGS,
                                             random_init=False,
                                             prune_rate=PRUNE_RATE,
                                             train_data_loader=train_data_loader,
                                             test_data_loader=test_data_loader,
                                             validation_iterations=VALIDATION_ITERATIONS,
                                             )
    accuracies_array_reinit = run_iterative_pruning(num_executions=NUM_EXECUTIONS,
                                                    num_prunings=NUM_PRUNINGS_REINIT,
                                                    random_init=True,
                                                    prune_rate=PRUNE_RATE,
                                                    train_data_loader=train_data_loader,
                                                    test_data_loader=test_data_loader,
                                                    validation_iterations=VALIDATION_ITERATIONS)

    # Plot figure 3
    plot_data = {}
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for pm, color in zip(PM_LIST, colors):
        key = f"{(1 - PRUNE_RATE) ** pm * 100:.1f}"
        plot_data[key] = {}
        plot_data[key]["accuracy"] = accuracies_array[pm]
        plot_data[key]['color'] = color

    colors_reinit = ['pink', 'cyan']
    for pm, color in zip(PM_LIST_REINIT[1:], colors_reinit):
        key = f"{(1 - PRUNE_RATE) ** pm * 100:.1f} (reinit)"
        plot_data[key] = {}
        plot_data[key]["accuracy"] = accuracies_array_reinit[pm]
        plot_data[key]['color'] = color

    fig, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)
    plot_dict(plot_data, ['green', 'orange', 'blue'], VALIDATION_ITERATIONS, ax1)
    plot_dict(plot_data, ['green', 'red', 'blue', 'purple', 'brown'], VALIDATION_ITERATIONS, ax2)
    plot_dict(plot_data, ['green', 'orange', 'blue', 'pink', 'cyan'], VALIDATION_ITERATIONS, ax3)

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
