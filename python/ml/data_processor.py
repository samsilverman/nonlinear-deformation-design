from __future__ import annotations
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from utils import InvertibleColumnTransformer



def get_parameters_processor() -> InvertibleColumnTransformer:
    """Data preprocessor for GCS design parameters.

    Returns
    -------
    processor : InvertibleColumnTransformer
        Data preprocessor for GCS design parameters.

    """
    processor = InvertibleColumnTransformer(transformers=[
        ('min-max', MinMaxScaler(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ('one-hot', OneHotEncoder(), [11]),
    ])

    return processor


def get_performance_processor() -> InvertibleColumnTransformer:
    """Data preprocessor for GCS uniaxial compression data.

    Returns
    -------
    processor : InvertibleColumnTransformer
        Data preprocessor for GCS uniaxial compression data.

    """
    processor = InvertibleColumnTransformer(transformers=[
        ('max-displacement', StandardScaler(), [0]),
        ('forces', Pipeline(steps=[
            ('log', FunctionTransformer(func=np.log, inverse_func=np.exp, validate=False)),
            ('standard', StandardScaler()),
            ('pca', PCA(n_components=10))]), slice(1, 101))
    ])

    return processor
