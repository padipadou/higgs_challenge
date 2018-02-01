# Descriptive analysis
#
# Authors : Paul-Alexis Dray,
#           Adrian Ahne
# Date : 01-02-2018
#
# Information: For testing just uncomment a piece of code
# We put everything in comments to avoid that everything is printed at the same time


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import sys

# so that the python packages who are installed in miniconda3 are seen (seaborn)
sys.path.append("/Users/adrianahne/miniconda3/envs/dataScienceEnv/lib/python3.5/site-packages")

# load data
data = pd.read_csv("training.csv")

# # check first rows of data
# print(data.head())

# # summary
# print(data.describe())

# # check if labels are balanced
# print(data.Label.value_counts())

# # print histograms of variables
# data.hist(bins=50, figsize=(20,20))
# plt.show()


# # ---- Plot covariance matrix -----

# # Compute the correlation matrix
# corr = data.corr()

# # Generate a mask for the upper triangle
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(10, 8))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 0, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
