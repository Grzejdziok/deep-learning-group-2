import argparse
import json
from typing import List, Tuple
import copy
import torch
import torch.nn as nn
import tqdm
import torchvision.datasets
import torch.nn.utils.prune as prune
import numpy as np
from model_factory import ModelFactory


def validate_model(data_loader: torch.utils.data.DataLoader,
                   model: nn.Module,
                   use_cuda: bool,
                   criterion: nn.Module,) -> Tuple[float, float]:
    num_hits = 0.
    running_loss = 0.
    dataset_count = 0
    for batch_input, batch_target in data_loader:
        dataset_count += batch_input.shape[0]
        if use_cuda:
            batch_input = batch_input.cuda()
            batch_target = batch_target.cuda()
        with torch.no_grad():
            batch_output_prob = model(batch_input)
            batch_output = torch.argmax(batch_output_prob, dim=1)
            num_hits += (batch_output == batch_target).sum().float().item()
            loss = criterion(batch_output_prob, batch_target)
        running_loss += loss.item() * batch_input.size(0)
    accuracy = num_hits / dataset_count
    loss = running_loss / dataset_count
    return accuracy, loss


def train_model(train_data_loader: torch.utils.data.DataLoader,
                test_data_loader: torch.utils.data.DataLoader,
                val_data_loader: torch.utils.data.DataLoader,
                USE_CUDA: bool,
                model: nn.Module,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                validate_at: np.ndarray,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iterations_so_far = 0
    iter_train_data_loader = iter(train_data_loader)
    total_iterations = max(validate_at)
    test_accuracies = []
    test_losses = []
    val_accuracies = []
    val_losses = []
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
            # Find accuracy and loss
            model.eval()
            test_accuracy, test_loss = validate_model(test_data_loader, model, USE_CUDA, criterion)
            val_accuracy, val_loss = validate_model(val_data_loader, model, USE_CUDA, criterion)
            test_accuracies.append(test_accuracy)
            val_accuracies.append(val_accuracy)
            test_losses.append(test_loss)
            val_losses.append(val_loss)
    progress_bar.close()
    test_accuracies = np.asarray(test_accuracies)
    test_losses = np.asarray(test_losses)
    val_accuracies = np.asarray(val_accuracies)
    val_losses = np.asarray(val_losses)
    return test_accuracies, test_losses, val_accuracies, val_losses


def count_parameters(model: nn.Module) -> int:
    num_parameters = 0
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            num_parameters += module.weight.data.numel()
            num_parameters += module.bias.data.numel()
    return num_parameters


def count_pruned_parameters(model: nn.Module) -> int:
    num_pruned_parameters = 0
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            if prune.is_pruned(module):
                num_pruned_parameters += module.weight_mask.data.numel() - module.weight_mask.data.count_nonzero().item()
    return num_pruned_parameters


def prune_model_l1(model: nn.Module, prune_ratio_hidden_fc: float, prune_ratio_hidden_conv: float, prune_ratio_output: float) -> nn.Module:
    assert hasattr(model, "hidden_layers") and hasattr(model, "output_layer")
    for module in model.hidden_layers:
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=prune_ratio_hidden_fc)
        elif isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=prune_ratio_hidden_conv)
    prune.l1_unstructured(model.output_layer, name="weight", amount=prune_ratio_output)
    return model


def prune_model_rnd(model: nn.Module, prune_ratio_hidden_fc: float, prune_ratio_hidden_conv: float, prune_ratio_output: float) -> nn.Module:
    assert hasattr(model, "hidden_layers") and hasattr(model, "output_layer")
    for module in model.hidden_layers:
        if isinstance(module, nn.Linear):
            prune.random_unstructured(module, name="weight", amount=prune_ratio_hidden_fc)
        elif isinstance(module, nn.Conv2d):
            prune.random_unstructured(module, name="weight", amount=prune_ratio_hidden_conv)
    prune.l1_unstructured(model.output_layer, name="weight", amount=prune_ratio_output)
    return model


def load_original_weights(model: nn.Module, model_original: nn.Module) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            if prune.is_pruned(module):
                mask = module.weight_mask
                prune.remove(module, 'weight')
                module.weight.data = model_original.get_submodule(
                    name).weight.data.clone()
                module.bias.data = model_original.get_submodule(
                    name).bias.data.clone()
                prune.custom_from_mask(module, 'weight', mask)
            else:
                module.weight.data = model_original.get_submodule(
                    name).weight.data.clone()
                module.bias.data = model_original.get_submodule(
                    name).bias.data.clone()
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
        model_factory: ModelFactory,
        num_executions: int,
        num_prunings: int,
        random_init: bool,
        learning_rate: float,
        prune_rate_fc: float,
        prune_rate_conv: float,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader,
        validation_iterations: np.ndarray,
        l1: bool,
        pm_list: List[int],
        file_name: str,
):
    test_accuracies_array = np.zeros(
        (num_prunings+1, num_executions, validation_iterations.shape[0]))
    test_losses_array = np.zeros(
        (num_prunings+1, num_executions, validation_iterations.shape[0]))
    val_accuracies_array = np.zeros(
        (num_prunings+1, num_executions, validation_iterations.shape[0]))
    val_losses_array = np.zeros(
        (num_prunings+1, num_executions, validation_iterations.shape[0]))
    weights_remaining_rates = []
    for i in range(num_executions):
        print(f"---ITERATION: {i + 1}, RANDOM_INIT={random_init}---")
        model = model_factory.create()
        model = random_reinit(model)
        if USE_CUDA:
            model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate)

        # Save original weights at the beginning
        model_original = copy.deepcopy(model)
        for pm in range(num_prunings+1):
            pruned_parameters = count_pruned_parameters(model)
            all_parameters = count_parameters(model)
            active_parameters = all_parameters - pruned_parameters
            current_prune_rate = pruned_parameters / all_parameters
            weights_remaining_rate = 1-current_prune_rate
            weights_remaining_rates.append(weights_remaining_rate)

            print(f"Training at params ratio: {weights_remaining_rate:.3f}, active parameters: {active_parameters}/{all_parameters}")
            test_accuracies, test_losses, val_accuracies, val_losses = \
                train_model(train_data_loader, val_data_loader, test_data_loader, USE_CUDA, model, criterion, optimizer, validation_iterations)
            if l1:
                model = prune_model_l1(
                    model=model,
                    prune_ratio_hidden_fc=prune_rate_fc,
                    prune_ratio_hidden_conv=prune_rate_conv,
                    prune_ratio_output=prune_rate_fc / 2,
                )
            else:
                model = prune_model_rnd(
                    model=model,
                    prune_ratio_hidden_fc=prune_rate_fc,
                    prune_ratio_hidden_conv=prune_rate_conv,
                    prune_ratio_output=prune_rate_fc / 2,
                )

            if not random_init:
                model = load_original_weights(
                    model=model,
                    model_original=model_original,
                )
            else:
                model = random_reinit(model)
            test_accuracies_array[pm, i, :] = test_accuracies
            test_losses_array[pm, i, :] = test_losses
            val_accuracies_array[pm, i, :] = val_accuracies
            val_losses_array[pm, i, :] = val_losses

            export_dict = {"VALIDATION_ITERATIONS": validation_iterations.tolist(),
                           "PM_LIST": pm_list,
                           "PRUNE_RATE": prune_rate_fc,
                           "weights_remaining_rates": weights_remaining_rates,
                           "test_accuracies": test_accuracies_array.tolist(),
                           "test_losses": test_losses_array.tolist(),
                           "val_accuracies": val_accuracies_array.tolist(),
                           "val_losses": val_losses_array.tolist(),
                           }
            with open(f"{file_name}.json", 'w') as file:
                json.dump(export_dict, file)


DATASET_NORMALIZATION = {
    "mnist": (0.1307, 0.3081),
    "fashion_mnist": (0.286, 0.353),
    "cifar10": ((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
}

DATASET_FACTORY = {
    "mnist": torchvision.datasets.MNIST,
    "fashion_mnist": torchvision.datasets.FashionMNIST,
    "cifar10": torchvision.datasets.CIFAR10,
}


if __name__ == "__main__":
    """
    Example commands:
    python main.py --dataset mnist --model lenet300100 --max-iteration 50000 --validate-each 100 --num-repetitions 1 --batch-size 60 --learning-rate 1.2e-3 --prune-rate-fc 0.2 --prune-rate-conv 0.0 --val-set-size 5000
    python main.py --dataset cifar10 --model conv2 --max-iteration 20000 --validate-each 100 --num-repetitions 1 --batch-size 60 --learning-rate 2e-4 --prune-rate-fc 0.2 --prune-rate-conv 0.1 --val-set-size 5000
    python main.py --dataset cifar10 --model conv4 --max-iteration 25000 --validate-each 100 --num-repetitions 1 --batch-size 60 --learning-rate 3e-4 --prune-rate-fc 0.2 --prune-rate-conv 0.1 --val-set-size 5000
    python main.py --dataset cifar10 --model conv6 --max-iteration 30000 --validate-each 100 --num-repetitions 1 --batch-size 60 --learning-rate 3e-4 --prune-rate-fc 0.2 --prune-rate-conv 0.15 --val-set-size 5000
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10"], required=True)
    parser.add_argument("--model", choices=["lenet300100", "conv2", "conv4", "conv6"], required=True)
    parser.add_argument("--max-iteration", type=int, required=True)
    parser.add_argument("--validate-each", type=int, required=True, default=100)
    parser.add_argument("--num-repetitions", type=int, required=True, default=1)
    parser.add_argument("--prune-rate-fc", type=float, required=True)
    parser.add_argument("--prune-rate-conv", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--val-set-size", type=int, default=5000, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    args = parser.parse_args()

    model_factory = ModelFactory(model_name=args.model)

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    validation_iterations = np.arange(args.validate_each, args.max_iteration + 1, args.validate_each, dtype=int)
    # P_m's from figure 3 - these are the exponents of 0.8 to get to roughly the Pm's for figure 3
    PM_LIST = [0, 3, 7, 12, 15, 18, 28]
    PM_LIST_REINIT = [0, 3, 7]

    num_prunings = max(PM_LIST)
    num_prunings_reinit = max(PM_LIST_REINIT)
    num_executions = args.num_repetitions
    prune_rate_conv = args.prune_rate_conv
    prune_rate_fc = args.prune_rate_fc

    USE_CUDA = torch.cuda.is_available()
    # MNIST statistics
    mean, std = DATASET_NORMALIZATION[args.dataset]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])
    dataset_factory = DATASET_FACTORY[args.dataset]
    train_val_dataset = dataset_factory(root="./.mnist", transform=transform, download=True, train=True)
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, (len(train_val_dataset)-5000, 5000))
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = dataset_factory(
        root="./.mnist", transform=transform, download=True, train=False)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    run_iterative_pruning(model_factory=model_factory,
                          num_executions=num_executions,
                          num_prunings=num_prunings,
                          random_init=False,
                          learning_rate=learning_rate,
                          prune_rate_fc=prune_rate_fc,
                          prune_rate_conv=prune_rate_conv,
                          train_data_loader=train_data_loader,
                          val_data_loader=val_data_loader,
                          test_data_loader=test_data_loader,
                          validation_iterations=validation_iterations,
                          l1=True,
                          pm_list=PM_LIST,
                          file_name='data',
                          )
    run_iterative_pruning(model_factory=model_factory,
                          num_executions=num_executions,
                          num_prunings=num_prunings,
                          random_init=True,
                          learning_rate=learning_rate,
                          prune_rate_fc=prune_rate_fc,
                          prune_rate_conv=prune_rate_conv,
                          train_data_loader=train_data_loader,
                          val_data_loader=val_data_loader,
                          test_data_loader=test_data_loader,
                          validation_iterations=validation_iterations,
                          l1=False,
                          pm_list=PM_LIST,
                          file_name='data_random',
                          )
    run_iterative_pruning(model_factory=model_factory,
                          num_executions=num_executions,
                          num_prunings=num_prunings_reinit,
                          random_init=True,
                          learning_rate=learning_rate,
                          prune_rate_fc=prune_rate_fc,
                          prune_rate_conv=prune_rate_conv,
                          train_data_loader=train_data_loader,
                          val_data_loader=val_data_loader,
                          test_data_loader=test_data_loader,
                          validation_iterations=validation_iterations,
                          l1=True,
                          pm_list=PM_LIST_REINIT,
                          file_name='data_reinit',
                          )
