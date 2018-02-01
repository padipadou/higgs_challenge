# Determine best model with TPOT (needs a lot of computation power)
#
# Authors : Paul-Alexis Dray,
#           Adrian Ahne
# Date : 01-02-2018
#
# Information: For testing just uncomment a piece of code
# We put everything in comments to avoid that everything is printed at the same time


import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import sys
from grid_search_pipeline import load_data

# so that the python packages who are installed in miniconda3 are seen (tpot)
sys.path.append("/Users/adrianahne/miniconda3/envs/dataScienceEnv/lib/python3.5/site-packages")

# -- Load data
X, y = load_data()

# -- Replace -999 by numpys NaN
X = X.replace(-999, np.NaN)

# -- Drop columns with too many missing values
X = X.drop(columns=['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet',
                    'DER_lep_eta_centrality', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta',
                    'PRI_jet_subleading_phi'])

# -- Fill NaN with mean
X.fillna(X.mean(), inplace=True)

X = X.values

# -- Train test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

# -- Searches best hyperparameters with TPOT
# n_jobs = 4 : using four cores (we had an external server)
# max_eval_time_mins = 0.5 : evaluates each single pipeline only 30 seconds (if not it takes too much time)
my_tpot = TPOTClassifier(verbosity=3, max_time_mins=600, cv=5, generations=10, n_jobs=2, random_state=0,
                         max_eval_time_mins=0.5,
                         periodic_checkpoint_folder="./best_models")

my_tpot.fit(X_train, y_train)

print(my_tpot.score(X_test, y_test))
