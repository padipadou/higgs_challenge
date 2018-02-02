# higgs\_challenge
The goal is to explore the potential of advanced ML methods to improve the discovery significance of the experiment. No knowledge of particle physics is required. Using simulated data with features characterizing events detected by ATLAS, your task is to classify events into "tau tau decay of a Higgs boson" versus "background."

# 1. Files
* **data**:

"training.csv"

* **code**:

"descriptive\_analysis.py"

"grid\_search\_pipeline.py"

"time\_measurement\_best\_models.py"

"tpot\_test.py"

"keras\_test.py"

"utility.py"

* **documentation**:

"higgs\_challenge\_report.pdf": **REPORT ABOUT OUR WORK**

"HiggsBosonMLChallenge.pdf"

"README.md"

* **others**:

directory "models" for neural networks from *Keras*

"requirements.txt" for required packages

# 2. Required packages
* On error type: "ImportError: No module named 'module\_name'"

`sudo pip3 install -r requirements.txt`

* On error type: "ImportError: No module named 'grid\_search\_pipeline'" or "ImportError: No module named 'utility'"

`export PYTHONPATH=$PYTHONPATH:/path_from_root_to_directory/higgs_challenge`

# 3. Run something
* **descriptive\_analysis.py**

`python3 descriptive_analysis.py`

First steps to study the data.

*Information*: For testing just uncomment a piece of code

We put everything in comments to avoid that everything is printed at the same time

* **grid\_search\_pipeline.py**

`python3 grid_search_pipeline.py`

Gridsearch using a preprocessing pipeline and one model can be chosen.

*Information*: When choosing one model to test, adapt the grid search parameter for this model

*Information*: For testing just uncomment a piece of code

We put everything in comments to avoid that everything is printed at the same time

* **time\_measurement\_best\_models.py**

`python3 time_measurement_best_models.py`

Measures the execution time of the training for the models with their 'best' configuration obtained by the gridsearch.

*Information*: For testing just uncomment a piece of code

We put everything in comments to avoid that everything is printed at the same time

* **tpot\_test.py**

`python3 tpot_test.py`

Using the new library tpot, which is automatically determining best model and hyperparameters

*Information*: For testing just uncomment a piece of code

We put everything in comments to avoid that everything is printed at the same time

* **keras\_test.py**

`python3 keras_test.py`

Applying neural networks

*Information*: For testing just uncomment a piece of code

We put everything in comments to avoid that everything is printed at the same time
