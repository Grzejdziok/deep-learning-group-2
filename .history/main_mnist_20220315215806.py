import torch
import torch.nn as nn
import torchvision
import torchvision.datasets
import torch.nn.utils.prune as prune

from lenet import Lenet300100


if __name__ == "__main__":

    BATCH_SIZE = 60
    LEARNING_RATE = 1.2e-3
    TOTAL_ITERATIONS = 10000
    PM = [100, 51.3, 21.1, 7.0, 3.6, 1.9] #P_m's from figure 3
    USE_CUDA = torch.cuda.is_available()

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
    while iterations_so_far < TOTAL_ITERATIONS:
        try:
            batch_input, batch_target = next(iter_train_data_loader)
            if USE_CUDA:
                batch_input = batch_input.cuda()
                batch_target = batch_target.cuda()
        except StopIteration:
            iter_train_data_loader = iter(train_data_loader)
            continue

        model.train()
        #Global pruning, not sure if its the one they use 

        print(model.layers[0])


        # parameters_to_prune = (     
        #     (model.conv1, 'weight'),
        #     (model.conv2, 'weight'),
        #     (model.fc1, 'weight'),
        #     (model.fc2, 'weight'),
        #     (model.fc3, 'weight'),
        # )

        # prune.global_unstructured(
        #     parameters_to_prune,
        #     pruning_method=prune.L1Unstructured,
        #     amount=0.2,
        # )

        batch_output_prob = model(batch_input)

        loss = criterion(batch_output_prob, batch_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        iterations_so_far += 1

        if iterations_so_far % 200 == 0:
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
            print(f"{iterations_so_far} ACCURACY: {accuracy}")
