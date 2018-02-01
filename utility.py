# Utility functions and classes that are used in the classification
#
# Authors : Paul-Alexis Dray,
#           Adrian Ahne
# Date : 01-02-2018
#


import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class MissingValues(BaseEstimator, TransformerMixin):
    """ Replaces missing values by numpys NaN functions
        Drops features with more than 70% of missing value_counts
        Replaces NaN's with mean/median
    """

    def transform(self, X, y=None, **fit_params):
        # import ipdb; ipdb.set_trace()

        assert isinstance(X, pd.DataFrame)

        # replace missing values
        X = X.replace(-999, np.NaN)

        # drop features with 177457 missing values of total 250000
        X = X.drop(columns=['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet',
                            'DER_lep_eta_centrality', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta',
                            'PRI_jet_subleading_phi'])

        # replace missing values with mean
        X.fillna(X.mean(), inplace=True)  # alternatively replace by median for example

        return X

    def fit(self, X, y=None, **fit_params):
        return self


class TypeSelector(BaseEstimator, TransformerMixin):
    """ Column selection after data type (boolean, numeric, categorical) """

    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class StringIndexer(BaseEstimator, TransformerMixin):
    """ Simple Indexer as OneHotEncoder only accepts positive values"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
        ))


class Debug(BaseEstimator, TransformerMixin):
    """ Debug function printing the head and shape of a pandas dataframe"""

    def transform(self, X):
        import ipdb;
        ipdb.set_trace()
        print(pd.DataFrame(X).head())
        print(X.shape)
        return X

    def fit(self, X, y=None, **fit_params):
        return self
