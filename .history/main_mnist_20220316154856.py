from matplotlib.pyplot import plot
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets
import torch.nn.utils.prune as prune
import numpy as np
import copy

from lenet import Lenet300100


if __name__ == "__main__":

    BATCH_SIZE = 60
    LEARNING_RATE = 1.2e-3
    TOTAL_ITERATIONS = np.arange(100, 50001, 100)
    PM = [0, 3, 7, 12, 15, 18] #P_m's from figure 3 - these are the powers of 0.8 to get to roughly the Pm's for figure 3
    plot_data = np.zeros((len(PM), len(TOTAL_ITERATIONS)))
    PRUNE_RATE = 0.2
    USE_CUDA = torch.cuda.is_available()
    USE_CUDA = False

    model = Lenet300100()
    if USE_CUDA:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.1/1.275, std=1./1.275),  # seemingly the normalization parameters according to LeCun (1998)
        torchvision.transforms.Lambda(lambda tensor: tensor.reshape(-1)),
    ])
    train_dataset = torchvision.datasets.MNIST(root="./.mnist", transform=transform, download=True, train=True)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root="./.mnist", transform=transform, download=True, train=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # What are iterations exactly? The paper does not say, but it seems that iteration == batch
    iterations_so_far = 0
    iter_train_data_loader = iter(train_data_loader)
    for total_iterations in TOTAL_ITERATIONS:       #These are the x-axis points where the data is taken    
        weights_original = []                       #Save original weights at the beginning
        for i in [0,2,4]:
            weights_original.append(copy.deepcopy(model.layers[i].weight))
        for pm in PM:                               #These are for the colours of different fig 3 lines
            for i in range(pm):                     #This is for how many times you need to prune and train, per fig 3 line
                #Train
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
                #Prune
                prune.random_unstructured(model.layers[0], name="weight", amount=PRUNE_RATE)
                prune.random_unstructured(model.layers[2], name="weight", amount=PRUNE_RATE)
                prune.random_unstructured(model.layers[4], name="weight", amount=PRUNE_RATE/2) #Last layer at half the rate - page 22
                #Update
                mask = list(model.named_buffers())
                for i in range(3):
                    weights_original[i]=weights_original[i]*mask[i][1]
                    model.layers[2*i].weight = weights_original[i]
                # print(model.layers[0].weight.count_nonzero())
            #Train again
            iterations_so_far = 0
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
                print(model.layers[0].weight.count_nonzero())
                iterations_so_far += 1

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
            print(f"Iterations: {total_iterations}, PM: {round(0.8**pm*100,1)}%, ACCURACY: {accuracy}")

        model = Lenet300100() #reinit

    
