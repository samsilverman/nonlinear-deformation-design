# Data

In a [comprehensive prior study](https://doi.org/10.1038/s41467-024-48534-4), we conducted uniaxial compression testing on **generalized cylindrical shells (GCS)** to explore their energy-absorbing capabilities.
This folder hosts the processed results of 12,705 experiments.
Each experiment includes:

1. **`ID_Number`**: A unique experiment identifier.

2. **GCS design parameters**: 12 design parameters. See the [GCS repository](https://github.com/bu-shapelab/gcs) for detailed descriptions of each parameter.

3. **Force-displacement curve**: 100-point force-displacement curve obtained from the uniaxial compression test.

## Files

| File | Description |
| - | - |
| `displacements.csv` | The maximum displacement (mm) for each sample. The 100 intermediate values are obtained by linear interpolation from 0 mm to this maximum. |
| `forces.csv` | The 100 force (N) values aligned with the displacement values from `displacements.csv`. |
| `parameters.csv` | The 12 design parameters for every GCS. |
