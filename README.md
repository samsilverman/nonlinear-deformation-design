# Data-Driven Nonlinear Deformation Design of 3D-Printable Shells

[![Read the Paper](https://img.shields.io/static/v1.svg?label=DOI&message=10.1089/3dp.2024.0175&color=blue)](https://doi.org/10.1089/3dp.2024.0175)

![Teaser](https://github.com/samsilverman/nonlinear-deformation-design/blob/main/assets/images/teaser.png)

This repository contains the code for our 3D Printing and Additive Manufacturing 2025 paper “Data-Driven Nonlinear Deformation Design of 3D-Printable Shells.”

   > Samuel Silverman, Kelsey L. Snapp, Keith A. Brown, Emily Whiting  
   > [*Data-Driven Nonlinear Deformation Design of 3D-Printable Shells*](https://sam-silverman.com/assets/pdf/Silverman-DataDrivenNonlinear.pdf)  
   > 3D Printing and Additive Manufacturing (2025)

## Data

In a [comprehensive prior study](https://doi.org/10.1038/s41467-024-48534-4), we conducted uniaxial compression testing on [generalized cylindrical shells (GCS)](https://github.com/bu-shapelab/gcs) to explore their energy-absorbing capabilities.

`data/` contains the processed results of 12,705 experiments:

- `displacements.csv`: The maximum displacement (mm) for each sample. The 100 intermediate values are obtained by linear interpolation from 0 mm to this maximum.
- `forces.csv`: The 100 force (N) values aligned with the displacement values from `displacements.csv`.
- `parameters.csv`: The 12 design parameters for each GCS sample.

## Getting Started

Clone the repository and create the provided Conda environment for convenience and reproducibility.
Then install the package locally so the command-line tools can import `nonlinear_deformation_design` directly.

```bash
git clone https://github.com/samsilverman/nonlinear-deformation-design.git
cd nonlinear-deformation-design
conda env create -f environment.yml
conda activate nonlinear-deformation-design
pip install -e .
```

## Code

The codebase is organized into two parts:

- `src/nonlinear_deformation_design/`: the main Python package
- `tools/`: command-line entry points

| Script | Description | Image |
| - | - | - |
| `train_forward.py` | Train the forward model. | |
| `train_tandem.py` | Train the tandem model using a pretrained forward model from `train_forward.py`. | |
| `test_forward.py` | Test the trained forward model. | |
| `test_tandem.py` | Test the trained tandem model. | |
| `forward_design.py` | Predict force-displacement curves for GCS designs. | ![forward_design.py screenshot](https://github.com/samsilverman/nonlinear-deformation-design/blob/main/assets/images/forward_design.svg) |
| `inverse_design.py` | Predict GCS designs for force-displacement curves. | ![inverse_design.py screenshot](https://github.com/samsilverman/nonlinear-deformation-design/blob/main/assets/images/inverse_design.svg) |

> [!NOTE]
> Each tool can be run from the repository root.
> For info on a specific tool run:
>
> ```bash
> python tools/[TOOL].py --help
> ```

### CSV formats

The `forward_design.py` and `inverse_design.py` tools accept CSV input files.
Example input files can be found in `examples/forward_inputs.csv` and `examples/inverse_inputs.csv`.

## Maintainers

- [Sam Silverman](https://github.com/samsilverman/) - [sssilver@bu.edu](mailto:sssilver@bu.edu)

## Acknowledgements

The authors thank Adedire Adesiji for brainstorming, assistance in constructing applications, and photography;
Helena Gill, Xingjian Han, and Abinit Sati for their work on constructing and running the impact absorption application;
and Peter Yichen Chen for his discussions and input.

## Citation

```bibtex
@article{Silverman:2025:DataDrivenNonlinear,
author = {Silverman, Samuel and Snapp, Kelsey L. and Brown, Keith A. and Whiting, Emily},
title = {Data-Driven Nonlinear Deformation Design of 3D-Printable Shells},
year = {2026},
journal = {3D Printing and Additive Manufacturing},
volume = {13},
number = {1},
pages = {90-100},
url = {https://doi.org/10.1089/3dp.2024.0175},
doi = {10.1089/3dp.2024.0175},
}
```

## License

Released under the [MIT License](https://github.com/samsilverman/nonlinear-deformation-design/blob/main/LICENSE).
