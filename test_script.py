import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn import feature_selection
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.decomposition import PCA

from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

#from sklearn.linear_model import LogisticRegression
#import sklearn.neural_network


from methods_functions import *



check_missing_val(genedata)
#plotimportances(initial_rank(genedata, gleasonscores), genedata, 40, "Initial")

#split data into test and training data
gene_train, gene_test, gleason_train, gleason_test = train_test_split(genedata, gleasonscores, test_size=0.25)

#print(gene_train)#print(gene_test)#print(gleason_train)#print(gleason_test)

gene_train_filt, gleason_train = variance_filter(gene_train, gleason_train)
gene_train_filt, gleason_train = univariate_filter(gene_train_filt, gleason_train)
gene_train_filt, gleason_train = correlation_filter(gene_train_filt, gleason_train)
gene_train_filt, gleason_train = recursive_feature_elimination(gene_train_filt, gleason_train)
gene_train_filt, gleason_train = feature_select_from_model(gene_train_filt, gleason_train)
#PCA_analysis(gene_train_filt, gleason_train, 3)
#gene_train_filt, gleason_train = tree_based_selection(gene_train_filt, gleason_train)
#gene_train_filt, gleason_train = L1_based_select(gene_train_filt, gleason_train)

#cop and paste down here order:




base_rfc = RandomForestClassifier(n_estimators = 100) #, max_features=None)
base_rfc.fit(gene_train, gleason_train['Gleason'] )
base_rfc_predictions = base_rfc.predict(gene_test)
print("Base RFC accuracy: ", accuracy_score(gleason_test, base_rfc_predictions))
print("Base Num features: ", base_rfc.n_features_)
#print("Num classes: ", base_rfc.n_classes_)
#print("Num outputs: ", base_rfc.n_outputs_)

def filter_data():
    global gene_train
    global gene_test
    global features_selected
    print("\n=====================================================\n")
    features_selected = list(gene_train_filt.columns.values)
    print("Features selected: ")
    print(features_selected[1:5])
    gene_train = gene_train[features_selected]
    gene_test = gene_test[features_selected]
    print("\n=====================================================\n")

filter_data()

print("Train gene shape:    ", gene_train.shape)

def model_accuracy():
    print("\n=====================================================\n")
    #Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=8) #value from validation curve, check with hyperparameters
    rfc.fit(gene_train, gleason_train['Gleason'] )
    rfc_predictions = rfc.predict(gene_test)
    print("RFC accuracy: ", accuracy_score(gleason_test['Gleason'], rfc_predictions))
    print("Num features: ", rfc.n_features_)
    #print("Num classes: ", rfc.n_classes_)
    #print("Num outputs: ", rfc.n_outputs_)
    print("\n=====================================================\n")

model_accuracy()



print("Method Log")
print(methodlog)

visualise_accuracy_methodlog()



def val_curve_gen(gene_data, gleason_score):
    # Create range of values for parameter
    param_range = np.arange(1, 250, 2)
    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(RandomForestClassifier(),
        gene_data, gleason_score['Gleason'], param_name='n_estimators',
        param_range=param_range, cv=5, scoring='accuracy', n_jobs=-1)
    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="red")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="green")
    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")
    # Create plot
    plt.title("Validation Curve With Random Forest Classifier")
    plt.xlabel("Number Of Trees")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()

val_curve_gen(genedata, gleasonscores)

#val_curve_gen(gene_train, gleason_train)




#from literature
#features_selected = ["BTG2", "IGFBP3", "SIRT1", "MXI1", "FDPS"]

"""
#values from experimental hyperparameter search

{'bootstrap': True,
 'max_depth': 30,
 'max_features': 'sqrt',
 'min_samples_leaf': 1,
 'min_samples_split': 5,
 'n_estimators': 400}

{'bootstrap': True,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': 80,
 'max_features': 2,
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 3,
 'min_samples_split': 8,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 100,
 'n_jobs': None,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}
"""



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
