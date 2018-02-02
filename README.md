# higgs_challenge
The goal is to explore the potential of advanced ML methods to improve the discovery significance of the experiment. No knowledge of particle physics is required. Using simulated data with features characterizing events detected by ATLAS, your task is to classify events into "tau tau decay of a Higgs boson" versus "background." 

# 1. Required packages:
* On error type: "ImportError: No module named 'module\_name'"

`sudo pip3 install -r requirements.txt`

* On error type: "ImportError: No module named 'grid\_search\_pipeline'" or "ImportError: No module named 'utility'"

`export PYTHONPATH=$PYTHONPATH:/path_from_root_to_directory/higgs_challenge`

# 2. Run something:
## descriptive_analysis.py

`python3 descriptive_analysis.py`

*Information*: For testing just uncomment a piece of code

We put everything in comments to avoid that everything is printed at the same time

## grid_search_pipeline.py

`python3 grid_search_pipeline.py`

*Information*: For testing just uncomment a piece of code

We put everything in comments to avoid that everything is printed at the same time

## time_measurement_best_models.py

`python3 time_measurement_best_models.py`

*Information*: For testing just uncomment a piece of code

We put everything in comments to avoid that everything is printed at the same time

## tpot_test.py

`python3 tpot_test.py`

*Information*: For testing just uncomment a piece of code

We put everything in comments to avoid that everything is printed at the same time

## keras_test.py

`python3 keras_test.py`

*Information*: For testing just uncomment a piece of code

We put everything in comments to avoid that everything is printed at the same time
