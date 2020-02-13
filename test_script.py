import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import sklearn.neural_network
from sklearn.decomposition import PCA

from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

from methods_functions import *

check_missing_val(genedata)

#plotimportances(initial_rank(genedata, gleasonscores), genedata, 40, "Initial")


#split data into test and training data
gene_train, gene_test, gleason_train, gleason_test = train_test_split(genedata, gleasonscores, test_size=0.05)


gene_train_filt, gleason_train = variance_filter(gene_train, gleason_train)
gene_train_filt, gleason_train = univariate_filter(gene_train_filt, gleason_train)
gene_train_filt, gleason_train = correlation_filter(gene_train_filt, gleason_train)
gene_train_filt, gleason_train = recursive_feature_elimination(gene_train_filt, gleason_train)
gene_train_filt, gleason_train = feature_select_from_model(gene_train_filt, gleason_train)
#PCA_analysis(gene_train_filt, gleason_train, 3)
gene_train_filt, gleason_train = tree_based_selection(gene_train_filt, gleason_train)
gene_train_filt, gleason_train = L1_based_select(gene_train_filt, gleason_train)



def filter_data():
    global gene_train
    global gene_test
    print("\n=====================================================\n")
    features_selected = list(gene_train_filt.columns.values)
    print("Features selected: ")
    print(features_selected)
    gene_train = gene_train[features_selected]
    gene_test = gene_test[features_selected]
    print("\n=====================================================\n")

filter_data()

print(gene_train.shape)


def model_accuracy():
    print("\n=====================================================\n")
    #Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(gene_train, gleason_train['Gleason'])
    rfc_predictions = rfc.predict(gene_test)
    print("RFC: ", accuracy_score(gleason_test['Gleason'], rfc_predictions))
    print("\n=====================================================\n")

model_accuracy()


print("Method Log")
print(methodlog)


visualise_accuracy_methodlog()







"""
#It seems Classifier is better as Gleason score is an ordinal variable
#Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(gene_train, gleason_train['Gleason'])
rfr_predictions = rfr.predict(gene_test)
print("RFR: ", rfr_predictions)

#Just to test
#Logistic Regression
#lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=7500)
#lr.fit(gene_train, gleason_train['Gleason'])    #predictions = lr.predict(gene_test)
#print("LR:  ", accuracy_score(gleason_test['Gleason'], predictions))
"""
