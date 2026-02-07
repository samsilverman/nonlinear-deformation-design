#!/usr/bin/env python3
"""Script for TNN forward design network training.

"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from load_seeds import load_seeds
from ml import load_data, get_parameters_processor, get_performance_processor, TNN, WeightedMSELoss
from utils import set_seed, split_indices, print_model, Trainer


def main() -> None:
    seeds = load_seeds()

    for seed in seeds:
        # Setup
        set_seed(seed=seed)

        # Data preprocessing
        parameters, performance = load_data()

        train_indices, valid_indices, _ = split_indices(num_samples=parameters.shape[0],
                                                        percent_train=0.8,
                                                        percent_valid=0.1)

        parameters_train = parameters.iloc[train_indices]
        performance_train = performance.iloc[train_indices]
        parameters_valid = parameters.iloc[valid_indices]
        performance_valid = performance.iloc[valid_indices]

        parameters_processor = get_parameters_processor()
        performance_processor = get_performance_processor()

        parameters_train = parameters_processor.fit_transform(parameters_train)
        performance_train = performance_processor.fit_transform(performance_train)

        parameters_valid = parameters_processor.transform(parameters_valid)
        performance_valid = performance_processor.fit_transform(performance_valid)

        # Datasets
        train_dataset = TensorDataset(torch.from_numpy(parameters_train).to(dtype=torch.float32),
                                      torch.from_numpy(performance_train).to(dtype=torch.float32))

        valid_dataset = TensorDataset(torch.from_numpy(parameters_valid).to(dtype=torch.float32),
                                    torch.from_numpy(performance_valid).to(dtype=torch.float32))

        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=16,
                                shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset,
                                batch_size=16,
                                shuffle=True)

        # Model
        model = TNN().f()

        print_model(model)

        # Criterion and optimizer
        forces_processor = performance_processor.named_transformers_['forces']
        pca_step = forces_processor.named_steps['pca']
        weights = pca_step.explained_variance_
        weights /= np.sum(a=weights)
        weights = np.insert(arr=weights, obj=0, values=1)
        weights = torch.tensor(data=weights, dtype=torch.float32)

        criterion = WeightedMSELoss(weights=weights)
        optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-5)

        # Train
        save_file = Path(__file__).resolve().parent.parent / 'models' / f'{seed}.pt'

        trainer = Trainer(model=model,
                         criterion=criterion,
                         optimizer=optimizer,
                         train_loader=train_loader,
                         valid_loader=valid_loader,
                         epochs=500,
                         resume=False,
                         save_file=save_file)

        trainer.train(verbose=True)


if __name__ == '__main__':
    main()
