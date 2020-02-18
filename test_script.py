import numpy as np
import pandas as pd
import itertools

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn import feature_selection
from sklearn.model_selection import cross_validate, cross_val_score, learning_curve, validation_curve
from sklearn.decomposition import PCA

from sklearn.tree import export_graphviz, plot_tree, DecisionTreeClassifier
from subprocess import call
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix

#from sklearn.linear_model import LogisticRegression
#import sklearn.neural_network


from methods_functions import *


predict_null_accuracy(gleasonscores)
check_missing_val(genedata)
#plotimportances(initial_rank(genedata, gleasonscores), genedata, 40, "Initial")

#split data into test and training data
gene_train, gene_test, gleason_train, gleason_test = train_test_split(genedata, gleasonscores, test_size=0.20)

#print(gene_train)#print(gene_test)#print(gleason_train)#print(gleason_test)

#gene_train_filt, gleason_train = variance_filter(gene_train, gleason_train, var_thresh=0.0)
#gene_train_filt, gleason_train = univariate_filter(gene_train_filt, gleason_train, k_val=1000)
#gene_train_filt, gleason_train = correlation_filter(gene_train_filt, gleason_train, corr_thresh=0.75)
#gene_train_filt, gleason_train = corrlation_with_target(gene_train_filt, gleason_train, corr_thresh=0.1)
#gene_train_filt, gleason_train = recursive_feature_elimination(gene_train_filt, gleason_train, n_feat=None)
#gene_train_filt, gleason_train = feature_select_from_model(gene_train_filt, gleason_train, thresh=None)
#PCA_analysis(gene_train_filt, gleason_train, 3)
#gene_train_filt, gleason_train = tree_based_selection(gene_train_filt, gleason_train, thresh=None)
#gene_train_filt, gleason_train = L1_based_select(gene_train_filt, gleason_train, thresh=None)

#copy and paste down here order:


gene_train_filt, methodlog = basic_method_iteration(gene_train, gleason_train)
print(methodlog)



base_rfc = RandomForestClassifier(n_estimators = 100, random_state=42) #, max_features=None)
base_rfc.fit(gene_train, gleason_train['Gleason'] )
base_rfc_predictions = base_rfc.predict(gene_test)
print("Base RFC accuracy:   ", accuracy_score(gleason_test, base_rfc_predictions))
print("Base RFC class rep:")
print(classification_report(gleason_test, base_rfc_predictions, zero_division=1))

print("Cross validating")
rfc_scores = cross_val_score(base_rfc, genedata, gleasonscores['Gleason'], cv=5, verbose=True, scoring='accuracy')
print("RFC Scores:  ", rfc_scores)
print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

#print("Base RFC sensitivity: ", recall_score(gleason_test['Gleason'], base_rfc_predictions,  average='samples'))
print("Base Num features:   ", base_rfc.n_features_)
#print("Num classes:         ", base_rfc.n_classes_)
#print("Num outputs:         ", base_rfc.n_outputs_)
#vis_trees(base_rfc, "base")



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
    rfc = RandomForestClassifier(n_estimators=100, random_state=42) #value from validation curve, check with hyperparameters
    global rfc_fit
    rfc_fit = rfc.fit(gene_train, gleason_train['Gleason'])
    global rfc_predictions
    rfc_predictions = rfc_fit.predict(gene_test)
    print("RFC accuracy:    ", accuracy_score(gleason_test['Gleason'], rfc_predictions))
    print("RFC classification report:")
    print(classification_report(gleason_test['Gleason'], rfc_predictions, zero_division=1))

    print("Cross validating")
    rfc_scores = cross_val_score(rfc, genedata[features_selected], gleasonscores['Gleason'], cv=5, verbose=True, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    #print("RFC sensitivity: ", recall_score(gleason_test['Gleason'], rfc_predictions, average='samples'))
    print("Num features:    ", rfc.n_features_)
    #print("Num classes:     ", rfc.n_classes_)
    #print("Num outputs:     ", rfc.n_outputs_)
    #print(rfc_fit.base_estimator_)
    #print(rfc_fit.estimators_)
    print("\n=====================================================\n")

model_accuracy()


print("Method Log")
print(methodlog)


print("\n=====================================================\n")

print("Visualise cross validation scores methodlog\n")
visualise_accuracy_methodlog(methodlog)


print("Plotting confusion matrix\n")
plot_confusion_matrix(rfc_fit, genedata[features_selected], gleasonscores['Gleason'], normalize='true').ax_.set_title("Multilabel Confusion Matrix")
plt.show()
#multilabel_confusion_plot(rfc_fit, gleason_test['Gleason'], rfc_predictions) #redundent after update


print("Plotting validation curve\n")
val_curve_gen(genedata[features_selected], gleasonscores)


print("Plotting learning curve\n")
plot_learning_curve(RandomForestClassifier(), genedata[features_selected], gleasonscores)

"""
if len(features_selected) < 40:
    print("Pairplot\n")
    mergedset = pd.merge(genedata[features_selected], gleasonscores['Gleason'], left_index=True, right_index=True)
    print(mergedset)
    sns.pairplot(mergedset, hue='Gleason')
    plt.show()


print("Plotting tree\n")
print(rfc_fit.base_estimator_)
print(rfc_fit.base_estimator_.tree_)
plot_tree(RandomForestClassifier(n_estimators=100, random_state=42).fit(genedata, gleasonscores['Gleason']).base_estimator_)
plt.show()
"""

#print("Plotting decision trees\n")
#vis_trees(RandomForestClassifier(n_estimators=100, random_state=42).fit(gene_train, gleason_train['Gleason']), "fit")







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

#if control/cancer model implimented
#print("Plotting Precision Recall Curve")
#plot_precision_recall_curve(rfc_fit, genedata[features_selected], gleasonscores)
"""
