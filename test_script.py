import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import sklearn.neural_network
from sklearn.decomposition import PCA

from methods_functions import *

check_missing_val(genedata)
#initial_rank(genedata, gleasonscores)
plotimportances(initial_rank(genedata, gleasonscores), 40, "Initial")

genedata, gleasonscores, mergedset = variance_filter(genedata, gleasonscores)
genedata, gleasonscores, mergedset = univariate_filter(genedata, gleasonscores)
genedata, gleasonscores, mergedset = correlation_filter(genedata, gleasonscores, mergedset)
genedata, gleasonscores, mergedset = recursive_feature_elimination(genedata, gleasonscores)
genedata, gleasonscores, mergedset = feature_select_from_model(genedata, gleasonscores)
genedata, gleasonscores, mergedset = PCA_filter(genedata, gleasonscores)

print("Method Log")
print(methodlog)

visualise_accuracy_methodlog()
