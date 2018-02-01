# Time measurement of training for models with their best configuration
# and with a cross validation of 10 Stratified K Fold
#
# Authors : Paul-Alexis Dray,
#           Adrian Ahne
# Date : 01-02-2018
#
# Information: For testing just uncomment a piece of code
# We put everything in comments to avoid that everything is printed at the same time

import numpy as np
import time
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utility import MissingValues, TypeSelector, StringIndexer, Debug
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from grid_search_pipeline import load_data


if __name__ == "__main__":
    # -- Load data
    X, y = load_data()

    # -- Best model configurations from grid search
    # WARNING! Change also parameter SelectKBest in pipeline for each model!!
    model = LogisticRegression(penalty='l2', solver='lbfgs', tol=1e-2, random_state=0)
    # model = LinearDiscriminantAnalysis(n_components=1, solver='lsqr', tol=1e-2, shrinkage='auto')
    # model = DecisionTreeClassifier(criterion='entropy', splitter='best', presort=True, random_state=0)
    # model = RandomForestClassifier(n_estimators=15, criterion='entropy', max_features='auto', oob_score=False, random_state=0)

    # -- Make pipeline
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
                ('selectKbest', SelectKBest(f_regression, k=15)),  # for logistic regression, LDA
                #            ('selectKbest', SelectKBest(f_regression, k = 'all')), # for decision tree, random forest

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

    # -- Cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    accuracies = []  # collects accuracies of each fold
    confusion_mat = []  # collects confusion matrix of each fold

    start_time = time.time()
    for train, test in cv.split(X, y):
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y.iloc[train]
        y_test = y.iloc[test]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        cm = confusion_matrix(y_test, y_pred)
        cm = 100 * cm / cm.sum(axis=1)[:, np.newaxis]
        confusion_mat.append(cm)
        print('Accuracy: {:.15f}% '.format(accuracy))

    end_time = time.time()

    # -- Prints mean accuracy of all folds
    accuracies = np.asarray(accuracies) * 100
    print('CV Accuracy: {:.10f}% +/- {:.2f}%'.format(accuracies.mean(), accuracies.std()))

    # -- Prints mean confusion matrix of all folds
    confusion_mat = np.asarray(confusion_mat)
    print('Confusion matrix (mean):')
    print(np.mean(confusion_mat, axis=0))
    print('Confusion matrix:')
    print(cm)

    print("Execution time in seconds: ", end_time - start_time)
