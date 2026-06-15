from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer


class InvertibleColumnTransformer(ColumnTransformer):
    """Implementation of `sklearn.compose.ColumnTransformer` that supports
    `inverse_transform()`.

    """

    def inverse_transform(self, Xt: np.ndarray) -> np.ndarray:
        """Apply `inverse_transform` for each transformer, concatenate results.

        Parameters
        ----------
        Xt : (N, Mᵗ) numpy.ndarray
            Transformed data.

        Returns
        -------
        X : (N, M) np.ndarray
            Data in its original representation.

        """
        arrays = []
        for name, indices in self.output_indices_.items():
            transformer = self.named_transformers_.get(name, None)
            if transformer is None:
                continue

            data = Xt[:, indices.start:indices.stop]
            data = transformer.inverse_transform(data)

            arrays.append(data)

        X = np.concatenate(arrays, axis=1)

        return X
