from typing import List

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


def train_model(train_data_loader: torch.utils.data.DataLoader,
                test_data_loader: torch.utils.data.DataLoader,
                USE_CUDA: bool,
                model: nn.Module,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                validate_at: List[int],
                ) -> List[float]:
    iterations_so_far = 0
    # What are iterations exactly? The paper does not say, but it seems that iteration == batch
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
                    test_batch_output = torch.argmax(test_batch_output_prob, dim=1)
                    num_hits += (test_batch_output == test_batch_target).sum().float().item()
            accuracy = num_hits / len(test_dataset)
            accuracies.append(accuracy)
    return accuracies


if __name__ == "__main__":

    BATCH_SIZE = 60
    LEARNING_RATE = 1.2e-3
    VALIDATION_ITERATIONS = np.arange(100, 15000, 100, dtype=int)
    PM = [0, 3, 7, 12, 15, 18] #P_m's from figure 3 - these are the exponents of 0.8 to get to roughly the Pm's for figure 3
    plot_data = np.zeros((len(PM), len(VALIDATION_ITERATIONS)))
    PRUNE_RATE = 0.2
    USE_CUDA = torch.cuda.is_available()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.1/1.275, std=1./1.275),  # seemingly the normalization parameters according to LeCun (1998)
        torchvision.transforms.Lambda(lambda tensor: tensor.reshape(-1)),
    ])
    train_dataset = torchvision.datasets.MNIST(root="./.mnist", transform=transform, download=True, train=True)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root="./.mnist", transform=transform, download=True, train=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    for pm in range(max(PM)+1):  #Iterate over pruning rates
        print(f"Training at params ratio: {(1-PRUNE_RATE)**pm:.3f}, active parameters: {sum(w.data.count_nonzero().item() for w in weights_original)}", )
        accuracies = train_model(train_data_loader, test_data_loader, USE_CUDA, model, criterion, optimizer, VALIDATION_ITERATIONS)
        #Prune
        if isinstance(model, Lenet300100): #specific to Lenet
            prune.random_unstructured(model.layers[0], name="weight", amount=PRUNE_RATE)
            prune.random_unstructured(model.layers[2], name="weight", amount=PRUNE_RATE)
            prune.random_unstructured(model.layers[4], name="weight", amount=PRUNE_RATE/2) #Last layer at half the rate - page 22
        else:
            raise NotImplementedError
        #Update
        mask = list(model.named_buffers())
        if isinstance(model, Lenet300100):  #specific to Lenet
            for i in range(3):
                weights_original[i]=weights_original[i]*mask[i][1]
                model.layers[2*i].weight = weights_original[i]
        else:
            raise NotImplementedError

        if pm in PM:
            plot_data[PM.index(pm), :] = accuracies

    #Plot figure 3
    colours = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    fig, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)
    for line in range(3):
        ax1.plot(VALIDATION_ITERATIONS, plot_data[line], label = f"{round(0.8**PM[line]*100,1)}%", color=colours[line])
    for line in [0,2,3,4,5]:
        ax2.plot(VALIDATION_ITERATIONS, plot_data[line], label = f"{round(0.8**PM[line]*100,1)}%", color=colours[line])
    for line in range(3):
        ax3.plot(VALIDATION_ITERATIONS, plot_data[line], label = f"{round(0.8**PM[line]*100,1)}%", color=colours[line])

    lines = []
    labels = []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)  
    legend = {label:line for label,line in zip(labels, lines)}
    fig.legend(list(legend.values()), list(legend.keys()), loc='upper center', frameon=False, ncol = len(labels))
    plt.show()
