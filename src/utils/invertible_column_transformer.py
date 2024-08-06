from __future__ import annotations

from sklearn.compose import ColumnTransformer
import numpy as np


class InvertibleColumnTransformer(ColumnTransformer):
    """Implementation of ``sklearn.compose.ColumnTransformer`` that supports
    ``inverse_transform()``.

    """

    def inverse_transform(self, Xt: np.ndarray) -> np.ndarray:
        """Apply ``inverse_transform`` for each transformer, concatenate results.

        Parameters
        ----------
        Xt : (n_samples, n_transformed_features) np.ndarray
            The transformed data.

        Returns
        -------
        X : (n_samples, n_features) np.ndarray
            The data in its original representation.

        """
        arrays = []
        for name, indices in self.output_indices_.items():
            transformer = self.named_transformers_.get(name, None)
            if transformer is None:
                continue

            data = Xt[:, indices.start: indices.stop]
            data = transformer.inverse_transform(data)

            arrays.append(data)

        X = np.concatenate(arrays, axis=1)

        return X
