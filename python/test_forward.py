#!/usr/bin/env python3
"""Script for TNN forward design network testing.

"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from load_seeds import load_seeds
from ml import load_data, get_parameters_processor, get_performance_processor, TNN, WeightedMSELoss, confidence_interval
from utils import set_seed, split_indices, Tester, load_state_dict


def main() -> None:
    seeds = load_seeds()

    losses = []

    for seed in seeds:
        print(f'seed: {seed}')

        # Setup
        set_seed(seed=seed)

        # Data preprocessing
        parameters, performance = load_data()

        train_indices, _, test_indices = split_indices(num_samples=parameters.shape[0],
                                                        percent_train=0.8,
                                                        percent_valid=0.1)

        parameters_train = parameters.iloc[train_indices]
        performance_train = performance.iloc[train_indices]
        parameters_test = parameters.iloc[test_indices]
        performance_test = performance.iloc[test_indices]

        parameters_processor = get_parameters_processor()
        performance_processor = get_performance_processor()

        parameters_processor.fit(parameters_train)
        performance_processor.fit(performance_train)

        parameters_test = parameters_processor.transform(parameters_test)
        performance_test = performance_processor.fit_transform(performance_test)

        # Datasets
        test_dataset = TensorDataset(torch.from_numpy(parameters_test).to(dtype=torch.float32),
                                     torch.from_numpy(performance_test).to(dtype=torch.float32))

        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=16,
                                shuffle=True)

        # Model
        model = TNN().f()

        model_dir = Path(__file__).resolve().parent.parent / 'models' / 'forward'
        load_state_dict(model=model, file=model_dir / f'{seed}.pt')

        # Criterion
        forces_processor = performance_processor.named_transformers_['forces']
        pca_step = forces_processor.named_steps['pca']
        weights = pca_step.explained_variance_
        weights /= np.sum(a=weights)
        weights = np.insert(arr=weights, obj=0, values=1)
        weights = torch.tensor(data=weights, dtype=torch.float32)

        criterion = torch.nn.MSELoss()
        # criterion = WeightedMSELoss(weights=weights)

        # Test
        tester = Tester(model=model,
                        criterion=criterion,
                        test_loader=test_loader)

        loss = tester.test(verbose=False)
        print(f'\tloss: {loss}')

        losses.append(loss)

    mean, interval = confidence_interval(losses)

    print(r'95% confidence interval:')
    print(f'\t{mean}±{interval}')

if __name__ == '__main__':
    main()
