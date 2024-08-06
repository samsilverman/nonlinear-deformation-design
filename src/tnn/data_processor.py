import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from utils import InvertibleColumnTransformer

parameters_processor = InvertibleColumnTransformer(transformers=[
    ('min-max', MinMaxScaler(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ('one-hot', OneHotEncoder(), [11]),
])

displacement_processor = StandardScaler()

forces_processor = Pipeline(steps=[
    ('log', FunctionTransformer(func=np.log, inverse_func=np.exp, validate=True)),
    ('standard', StandardScaler()),
    ('pca', PCA(n_components=10)),
])
