#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import time
from typing import TYPE_CHECKING

from nonlinear_deformation_design import DEFAULT_DTYPE, DEFAULT_DEVICE
from nonlinear_deformation_design import set_seed, load_data, split_indices, get_parameters_processor, get_performance_processor, TNN, WeightedMSELoss
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import torch
from torch.utils.data import TensorDataset, DataLoader

if TYPE_CHECKING:
    from argparse import Namespace

DTYPE = DEFAULT_DTYPE
DEVICE = DEFAULT_DEVICE


def calculate_work(performance: np.ndarray) -> np.ndarray:
    """Calculates the work for force-displacement curves.

    Parameters
    ----------
    performance : (N, 101) numpy.ndarray
        Uniaxial compression data.
        The first entry is the maximum displacement and
        the following 100 entries are the force data.

    Returns
    -------
    work : (N,) numpy.ndarray
        Work (J) for force-displacement curves.

    """
    max_displacements = performance[:, 0].reshape(-1, 1)
    displacements = max_displacements * np.linspace(start=0, stop=1, num=100)
    forces = performance[:, 1:]

    work = np.trapz(y=forces, x=displacements, axis=1)

    # mJ to J
    work /= 1000

    return work


def calculate_stiffness(performance: np.ndarray) -> np.ndarray:
    """Calculates the stiffness for force-displacement curves.

    Parameters
    ----------
    performance : (N, 101) numpy.ndarray
        Uniaxial compression data.
        The first entry is the maximum displacement and
        the following 100 entries are the force data.

    Returns
    -------
    stiffnesses : (N,) numpy.ndarray
        Stiffness (N/mm) for force-displacement curves.

    """
    max_displacements = performance[:, 0].reshape(-1, 1)
    displacements = max_displacements * np.linspace(start=0, stop=1, num=100)
    forces = performance[:, 1:]

    # only look at beginning 25% of curves
    x = np.array_split(ary=displacements, indices_or_sections=4, axis=1)[0]
    y = np.array_split(ary=forces, indices_or_sections=4, axis=1)[0]

    stiffness = []

    # Report the stiffness as the maximum slope for data windows of size 5
    window_size = 5

    for index in range(x.shape[0]):
        slopes = []
        for start in range(x.shape[1] - window_size):
            stop = start + window_size

            x_window = x[index, start:stop]
            y_window = y[index, start:stop]

            fit = np.polyfit(x=x_window, y=y_window, deg=1)

            slopes.append(fit[0])

        stiffness.append(np.max(slopes))

    stiffness = np.array(stiffness)

    return stiffness


def build_parser() -> ArgumentParser:
    """Command-line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line interface.

    """
    parser = ArgumentParser(description='Test forward model.', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--percent-train', type=float, default=0.8, help='Fraction of samples used for training during forward model training.')
    parser.add_argument('--percent-valid', type=float, default=0.1, help='Fraction of samples used for validation during forward model training.')
    parser.add_argument('--seed', type=int, default=100, help='RNG seed used during forward model training.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size.')
    parser.add_argument('--dir', type=Path, default=Path('results'), help='Directory containing the saved forward model.')

    return parser


def validate_cli_arguments(args: Namespace, parser: ArgumentParser) -> None:
    """Validate command-line interface (CLI) arguments.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments.
    parser : argparse.ArgumentParser
        Parser used to raise errors.

    """
    if args.percent_train <= 0 or args.percent_train > 1:
        parser.error(f'--percent-train must be in (0, 1], got {args.percent_train}.')

    if args.percent_valid <= 0 or args.percent_train + args.percent_valid > 1:
        parser.error(f'--percent-valid must be in (0, {1 - args.percent_train}], got {args.percent_valid}.')

    max_seed = 2**32 - 1
    if args.seed < 0 or args.seed > max_seed:
        parser.error(f'--seed must be in [0, {max_seed}], got {args.seed}.')

    if args.batch_size <= 0:
        parser.error(f'--batch-size must be positive, got {args.batch_size}.')


def main() -> None:
    """Test forward model from the command line.

    """
    #############################################
    ########## CLI argument validation ##########
    #############################################

    parser = build_parser()
    args = parser.parse_args()

    validate_cli_arguments(args=args, parser=parser)

    ###########################
    ########## Setup ##########
    ###########################

    set_seed(seed=args.seed)

    # Data preprocessing
    parameters, performance = load_data()

    train_indices, _, test_indices = split_indices(num_samples=parameters.shape[0],
                                                   percent_train=args.percent_train,
                                                   percent_valid=args.percent_valid)

    parameters_train = parameters[train_indices]
    performance_train = performance[train_indices]
    parameters_test = parameters[test_indices]
    performance_test = performance[test_indices]
    performance_test_raw = performance_test.copy()

    parameters_processor = get_parameters_processor()
    performance_processor = get_performance_processor()

    parameters_processor.fit(X=parameters_train)
    performance_processor.fit(X=performance_train)

    parameters_test = parameters_processor.transform(X=parameters_test)
    performance_test = performance_processor.transform(X=performance_test)

    # Datasets
    test_dataset = TensorDataset(torch.from_numpy(parameters_test).to(dtype=DTYPE),
                                 torch.from_numpy(performance_test).to(dtype=DTYPE))

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    # Model
    model = TNN().f().to(dtype=DTYPE, device=DEVICE)

    state_dict = torch.load(f=args.dir / 'forward-model.pt', map_location=DEVICE)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    # Criterion
    forces_processor = performance_processor.named_transformers_['forces']
    pca_step = forces_processor.named_steps['pca']

    weights = pca_step.explained_variance_
    weights /= np.sum(weights)
    weights = np.insert(arr=weights, obj=0, values=1)
    weights = torch.tensor(data=weights, dtype=DTYPE, device=DEVICE)

    criterion = WeightedMSELoss(weights=weights)

    #############################
    ########## Testing ##########
    #############################

    print(f'{"-" * 5}Testing start (device: {DEVICE}){"-" * 5}')

    start_time = time.time()

    running_loss = 0
    performance_outputs_all = []

    for batch in test_loader:
        parameters, performance_targets = batch

        parameters = parameters.to(device=DEVICE, non_blocking=True)
        performance_targets = performance_targets.to(device=DEVICE, non_blocking=True)

        with torch.set_grad_enabled(mode=False):
            performance_outputs = model(parameters)

        loss = criterion(performance_outputs, performance_targets)

        running_loss += loss.item() * parameters.size(0)
        performance_outputs_all.append(performance_outputs.cpu().numpy())

    epoch_loss = running_loss / len(test_loader.dataset)

    performance_outputs_all = np.vstack(performance_outputs_all)

    performance_outputs_all = performance_processor.inverse_transform(Xt=performance_outputs_all)

    stiffness_targets = calculate_stiffness(performance=performance_test_raw)
    stiffness_outputs = calculate_stiffness(performance=performance_outputs_all)

    stiffness_r2 = r2_score(y_true=stiffness_targets, y_pred=stiffness_outputs)
    stiffness_mae = mean_absolute_error(y_true=stiffness_targets, y_pred=stiffness_outputs)

    work_targets = calculate_work(performance=performance_test_raw)
    work_outputs = calculate_work(performance=performance_outputs_all)

    work_r2 = r2_score(y_true=work_targets, y_pred=work_outputs)
    work_mae = mean_absolute_error(y_true=work_targets, y_pred=work_outputs)

    max_displacement_r2 = r2_score(y_true=performance_test_raw[:, 0], y_pred=performance_outputs_all[:, 0])
    max_displacement_mae = mean_absolute_error(y_true=performance_test_raw[:, 0], y_pred=performance_outputs_all[:, 0])

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    milliseconds = int(elapsed_time % 1 * 1000)

    print(f'Test loss: {epoch_loss}')
    print(f'Stiffness r2: {stiffness_r2}')
    print(f'Stiffness mae: {stiffness_mae}')
    print(f'Work r2: {work_r2}')
    print(f'Work mae: {work_mae}')
    print(f'Max displacement r2: {max_displacement_r2}')
    print(f'Max displacement mae: {max_displacement_mae}')
    print(f'Time: {minutes:02}:{seconds:02}.{milliseconds:03}')
    print(f'{"-" * 5}Testing end{"-" * 5}')


if __name__ == '__main__':
    main()
