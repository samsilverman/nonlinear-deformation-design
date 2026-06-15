#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING

import gcs
from matplotlib import pyplot as plt
from nonlinear_deformation_design import DEFAULT_DEVICE, DEFAULT_DTYPE
from nonlinear_deformation_design import TNN, get_parameters_processor, get_performance_processor
from nonlinear_deformation_design import load_data, set_seed, split_indices
import numpy as np
import pandas as pd
import pyvista as pv
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
    parser = ArgumentParser(description='Predict GCS designs for force-displacement curves.',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--csv', type=Path, required=True, help='CSV file containing input force-displacement curve data. See README for required columns.')
    parser.add_argument('--dir', type=Path, default=Path('results'), help='Directory containing the saved tandem model.')
    parser.add_argument('--out-dir', type=Path, default=Path('results'), help='Output directory for the saved results.')
    parser.add_argument('--seed', type=int, default=100, help='RNG seed used during tandem model training.')
    parser.add_argument('--percent-train', type=float, default=0.8, help='Fraction of samples used for training during tandem model training.')
    parser.add_argument('--percent-valid', type=float, default=0.1, help='Fraction of samples used for validation during tandem model training.')

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


def mesh_image(file: Path) -> None:
        """GCS design rendered image.

        Parameters
        ----------
        file : pathlib.Path
            File. Must have `.stl` extension.

        """
        mesh = pv.read(filename=file, file_format='stl')

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh=mesh, color='#95c7ff', lighting=True)
        plotter.show(return_img=True)

        image = plotter.screenshot(transparent_background=False)
        
        return np.array(image)


def visualize_mesh(file: Path, material: str) -> None:
        """Visualize predicted GCS design.

        Parameters
        ----------
        file : pathlib.Path
            File. Must have `.stl` extension.
        material : str
            GCS design material parameter.

        """
        image = mesh_image(file=file)

        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(6.4, 4.8))

        ax.imshow(X=image)
        ax.set_axis_off()

        fig.suptitle(t='Predicted design')
        ax.set_title(label=f'(material: {material})')

        plt.savefig('inverse_design.svg', dpi=300)
        plt.savefig(file.with_suffix(suffix='.png'), dpi=300)
        plt.show()


def main() -> None:
    """Predict GCS designs for force-displacement curves from the command line.

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
    model = TNN().to(dtype=DTYPE, device=DEVICE)

    state_dict = torch.load(f=args.dir / 'tandem-model.pt', map_location=DEVICE)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    # Performance
    performance = pd.read_csv(args.csv, delimiter=',', header=0).to_numpy()
    displacements = performance[:, 0]
    forces = performance[:, 1]
    performance = np.concatenate(([displacements[-1]], forces)).reshape(1, 101)

    performance = performance_processor.transform(X=performance)
    performance = torch.tensor(data=performance, dtype=DTYPE, device=DEVICE)

    ###############################
    ########## Inference ##########
    ###############################

    with torch.no_grad():
        design, _ = model(performance)

    # Export to numpy and invert processing
    design = design.cpu().numpy()
    design = parameters_processor.inverse_transform(Xt=design)
    design = design.squeeze()

    # Mesh
    mesh = gcs.GCS(c4_base=design[0],
                   c8_base=design[1],
                   c4_top=design[2],
                   c8_top=design[3],
                   twist_linear=design[4],
                   twist_amplitude=design[5],
                   twist_cycles=design[6],
                   perimeter_ratio=design[7],
                   height=design[8],
                   mass=design[9],
                   thickness=design[10],
                   triangulate_faces=False)
    
    # Save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gcs.io.save_mesh(file=args.out_dir / 'predicted_design.stl', shape=mesh)

    columns = [
        'c4_base',
        'c8_base',
        'c4_top',
        'c8_top',
        'twist_linear',
        'twist_amplitude',
        'twist_period',
        'perimeter_ratio',
        'height',
        'mass',
        'thickness',
        'material'
    ]
    
    df = pd.DataFrame([design], columns=columns)
    df.to_csv(args.out_dir / 'predicted_design.csv', index=False, float_format='%.6f')

    ##############################
    ########## Plotting ##########
    ##############################

    visualize_mesh(file=args.out_dir / 'predicted_design.stl', material=design[11])


if __name__ == '__main__':
    main()
