#!/usr/bin/env python3
"""Script for TNN inverse design network training.

This script assumes that the forward design network has
already been trained.

"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from utils import set_seed, split_indices, load_state_dict, Trainer
from ml import load_seeds, load_data, get_parameters_processor, get_performance_processor, TNNDataset, TNN, WeightedMSELoss, LILoss


def main() -> None:
    seeds = load_seeds()

    parameters, performance = load_data()

    alphas = [0, 0.01, 0.1, 1]

    for seed in seeds:
        # Setup
        print(f'Seed: {seed}')

        set_seed(seed=seed)

        # Data preprocessing
        train_indices, valid_indices, _ = split_indices(num_samples=parameters.shape[0],
                                                        percent_train=0.8,
                                                        percent_valid=0.1)

        parameters_train = parameters[train_indices]
        performance_train = performance[train_indices]
        parameters_valid = parameters[valid_indices]
        performance_valid = performance[valid_indices]

        parameters_processor = get_parameters_processor()
        performance_processor = get_performance_processor()

        parameters_train = parameters_processor.fit_transform(parameters_train)
        performance_train = performance_processor.fit_transform(performance_train)

        parameters_valid = parameters_processor.transform(parameters_valid)
        performance_valid = performance_processor.transform(performance_valid)

        # Datasets
        train_dataset = TNNDataset(parameters=torch.from_numpy(parameters_train).to(torch.float32),
                                   performance=torch.from_numpy(performance_train).to(torch.float32))

        valid_dataset = TNNDataset(parameters=torch.from_numpy(parameters_valid).to(torch.float32),
                                   performance=torch.from_numpy(performance_valid).to(torch.float32))

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=16,
                                  shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=16,
                                  shuffle=True)

        for alpha in alphas:
            print(f'Alpha: {alpha}')

            # Model
            model = TNN()

            forward_save_file = Path(__file__).resolve().parent.parent / 'models' / 'forward' / f'{seed}.pt'

            load_state_dict(model=model.f_,
                            file=forward_save_file)
            model.freeze_f()

            # Criterion and optimizer
            forces_processor = performance_processor.named_transformers_['forces']
            pca_step = forces_processor.named_steps['pca']
            weights = pca_step.explained_variance_
            weights /= np.sum(weights)
            weights = np.insert(arr=weights,
                                obj=0,
                                values=1)
            weights = torch.tensor(data=weights, dtype=torch.float32)

            Lp = WeightedMSELoss(weights=weights)
            Ld = nn.MSELoss()

            criterion = LILoss(Lp=Lp, Ld=Ld, alpha=alpha)
            optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-5)

            # Train
            save_file = Path(__file__).resolve().parent.parent / 'models' / f'{seed}-{alpha}.pt'

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
