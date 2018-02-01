# Grid-search with our pipeline
#
# Authors : Paul-Alexis Dray,
#           Adrian Ahne
# Date : 01-02-2018
#
# Information: For testing just uncomment a piece of code
# We put everything in comments to avoid that everything is printed at the same time

import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utility import MissingValues, TypeSelector, StringIndexer, Debug
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

DATA_PATH = "training.csv"

# ------ custom metric function (NOT USED) -----------
# Make a custom metric function
# def ams(s, b):
#    return math.sqrt(2 * ((s + b + 10) * math.log(1.0 + s/(b + 10)) - s))
#
# def get_ams_score(y_true, y_pred, weight):
#    s = weight * (y_true == 1) * (y_pred == 1)
#    b = weight * (y_true == 0) * (y_pred == 1)
#    s = np.sum(s)
#    b = np.sum(b)
#    return ams(s, b)

# Make a custom a scorer from the custom metric function
# Note: greater_is_better=False in make_scorer below would mean that the scoring function should be minimized.

# my_custom_scorer = make_scorer(get_ams_score, weight=W, greater_is_better=True)

# -------- end custom metric function (NOT USED) -----------


def load_data(data_path=DATA_PATH):
    data = pd.read_csv(DATA_PATH)

    # -- Transform variable PRI_jet_num to type categorical, only 4 possible values
    data.PRI_jet_num = data.PRI_jet_num.astype('category')

    print("X-y splitting...")
    # -- Drop useless variables for classification
    X = data.drop(columns=['Label', 'EventId', 'Weight'])

    # -- Replace y features to binary
    y = data.Label.replace(to_replace=['s', 'b'], value=[1, 0])

    # -- Get weight for each sample
    # W = data.Weight.values ( wanted to use the variable Weight as the weight of each observation, did not work)

    return X, y


def make_pipeline(model=True):
    # -- Choose model to be tested in the pipeline
    model = LogisticRegression(random_state=0)
    # model = LinearDiscriminantAnalysis()
    # model = DecisionTreeClassifier(random_state=0)
    # model = RandomForestClassifier(random_state=0)

    # -- Building the pipeline (for all models the same)
    if model:
        pipeline = Pipeline([

        # handle missing values
        ('missing_values', MissingValues()),

        ('features', FeatureUnion(n_jobs=1, transformer_list=[

            # only for boolean variables (do not exist here, only for completeness)
            ('boolean', Pipeline([
                ('selector', TypeSelector('bool')),
                # ('debug_bool', Debug()),
            ])),

            # only for numerical values
            ('numericals', Pipeline([
                ('selector', TypeSelector(np.number)),
                ('scaler', StandardScaler()),
                ('selectKbest', SelectKBest(f_regression)),
                # ('debug_bool', Debug()),
            ])),  # numericals close

            # only for categorical values
            ('categoricals', Pipeline([
                ('selector', TypeSelector('category')),
                ('labeler', StringIndexer()),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
                # ('debug_bool', Debug()),
            ]))

        ])),

        # model to be applied
        ('model', model)
    ])
    else:
        pipeline = Pipeline([

            # handle missing values
            ('missing_values', MissingValues()),

            ('features', FeatureUnion(n_jobs=1, transformer_list=[

                # only for boolean variables (do not exist here, only for completeness)
                ('boolean', Pipeline([
                    ('selector', TypeSelector('bool')),
                    # ('debug_bool', Debug()),
                ])),

                # only for numerical values
                ('numericals', Pipeline([
                    ('selector', TypeSelector(np.number)),
                    ('scaler', StandardScaler()),
                    ('selectKbest', SelectKBest(f_regression)),
                    # ('debug_bool', Debug()),
                ])),  # numericals close

                # only for categorical values
                ('categoricals', Pipeline([
                    ('selector', TypeSelector('category')),
                    ('labeler', StringIndexer()),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
                    # ('debug_bool', Debug()),
                ]))

            ]))

        ])

    return pipeline


def auc_scorer(target_score, prediction):
    """ uses roc auc scorer """
    auc_value = roc_auc_score(prediction, target_score)
    return auc_value


if __name__ == "__main__":
    print("Load data...")
    X, y = load_data()

    print("Building the pipeline...")
    pipeline = make_pipeline()

    # -- Train-test split
    print("Train test splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)

    print("Building Grid search...")
    # -- AUC score for grid search
    scorer = make_scorer(auc_scorer, greater_is_better=True)

    # -- Options for grid search; Uncomment configurations for the chosen model
    param_grid = {
        # selectKbest features for all model!
        'features__numericals__selectKbest__k': [15, 20, 'all'],

        # Configurations LogisticRegression:
        'model__penalty': ['l2'],
        'model__solver': ['newton-cg', 'lbfgs'],
        'model__C': [1, 0.1, 0.01, 0.001, 0.0001],
        'model__tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

        # # Configuration LinearDiscriminantAnalysis:
        # 'model__n_components' : [1, 3, 7, 10, 13,],
        # 'model__solver' : ['svd'],
        # 'model__solver' : ['lsqr', 'svd'],
        # 'model__tol' : [1e-2, 1e-3, 1e-4, 1e-5],
        # 'model__shrinkage' : [0, 0.2, 0.4, 0.6, 0.8, 1.0, 'auto', None]

        # # Configuration DecisionTreeClassifier:
        # 'model__criterion' : ['gini', 'entropy'],
        # 'model__splitter' : ['best', 'random'],
        # 'model__presort' : [True, False]

        # # Configuration RandomForestClassifier:
        # 'model__n_estimators' : [5, 10, 15],
        # 'model__criterion' : ['gini', 'entropy'],
        # 'model__max_features' : ['auto', 'log2', None],
        # 'model__oob_score' : [True, False ]
    }

    # -- Grid search: (cross-validation 3 fold; n_jobs=4 (we had an external server and could use 4 cores))
    grid = GridSearchCV(pipeline, cv=3, param_grid=param_grid, n_jobs=1, verbose=2)  # 3k-fold
    # grid = GridSearchCV(pipeline, cv=3, param_grid=param_grid, n_jobs=4, verbose=3, scoring=scorer) # 3k-fold scoring=['roc_auc'],

    # -- Run grid search
    print("Running Grid search...")
    # grid.fit(X_train, y_train, **{'model__sample_weight': W}) # sample weight did not work
    grid.fit(X_train, y_train)

    # -- Summarize results
    print("\nBest: %f using %s" % (grid.best_score_, grid.best_params_))
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
