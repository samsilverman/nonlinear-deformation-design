#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from nonlinear_deformation_design import DEFAULT_DEVICE, DEFAULT_DTYPE
from nonlinear_deformation_design import TNN, get_parameters_processor, get_performance_processor
from nonlinear_deformation_design import load_data, set_seed, split_indices
import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from argparse import Namespace

DTYPE = DEFAULT_DTYPE
DEVICE = DEFAULT_DEVICE


def build_parser() -> ArgumentParser:
    """Command-line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line interface.

    """
    parser = ArgumentParser(description='Predict force-displacement curves for GCS designs.',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--csv', type=Path, required=True, help='CSV file containing input GCS parameter data. See README for required columns.')
    parser.add_argument('--dir', type=Path, default=Path('results'), help='Directory containing the saved forward model.')
    parser.add_argument('--out-dir', type=Path, default=Path('results'), help='Output directory for the saved results.')
    parser.add_argument('--seed', type=int, default=100, help='RNG seed used during forward model training.')
    parser.add_argument('--percent-train', type=float, default=0.8, help='Fraction of samples used for training during forward model training.')
    parser.add_argument('--percent-valid', type=float, default=0.1, help='Fraction of samples used for validation during forward model training.')

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
    max_seed = 2**32 - 1
    if args.seed < 0 or args.seed > max_seed:
        parser.error(f'--seed must be in [0, {max_seed}], got {args.seed}.')

    if args.percent_train <= 0 or args.percent_train > 1:
        parser.error(f'--percent-train must be in (0, 1], got {args.percent_train}.')

    if args.percent_valid <= 0 or args.percent_train + args.percent_valid > 1:
        parser.error(f'--percent-valid must be in (0, {1 - args.percent_train}], got {args.percent_valid}.')


def visualize_curve(displacements: np.ndarray, forces: np.ndarray, file: Path) -> None:
        """Visualize predicted force-displacement curve.

        Parameters
        ----------
        displacements : (100,) numpy.ndarray
            Predicted displacements.
        forces : (100,) numpy.ndarray
            Predicted forces.
        file : pathlib.Path
            File. Must have `.png` extension.

        """
        _, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(6.4, 4.8))

        ax.plot(displacements, forces)

        ax.set_xlabel('Displacement (mm)')
        ax.set_ylabel('Force (N)')
        ax.set_title(label='Predicted force-displacement curve')

        plt.savefig(file, dpi=300)
        plt.show()


def main() -> None:
    """Predict force-displacement curves for GCS designs from the command line.

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

    train_indices, _, _ = split_indices(num_samples=parameters.shape[0],
                                        percent_train=args.percent_train,
                                        percent_valid=args.percent_valid)

    parameters_train = parameters[train_indices]
    performance_train = performance[train_indices]

    parameters_processor = get_parameters_processor()
    performance_processor = get_performance_processor()

    parameters_processor.fit(X=parameters_train)
    performance_processor.fit(X=performance_train)

    # Model
    model = TNN().f().to(dtype=DTYPE, device=DEVICE)

    state_dict = torch.load(f=args.dir / 'forward-model.pt', map_location=DEVICE)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    # Design
    design = pd.read_csv(args.csv, delimiter=',', header=0).to_numpy()
    design = parameters_processor.transform(X=design)
    design = torch.tensor(data=design, dtype=DTYPE, device=DEVICE)

    ###############################
    ########## Inference ##########
    ###############################

    with torch.no_grad():
        performance = model(design)

    # Export to numpy and invert processing
    performance = performance.cpu().numpy()
    performance = performance_processor.inverse_transform(Xt=performance)
    performance = performance.squeeze()

    # Force-displacement data
    displacements = performance[0] * np.linspace(start=0, stop=1, num=100)
    forces = performance[1:]

    # Save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(fname=args.out_dir / 'predicted_performance.csv',
               X=np.column_stack((displacements, forces)),
               delimiter=',',
               header='Displacement (mm),Force (N)',
               comments='',
               fmt='%f')

    ##############################
    ########## Plotting ##########
    ##############################

    visualize_curve(displacements=displacements, forces=forces, file=args.out_dir / 'predicted_performance.png')


if __name__ == '__main__':
    main()
