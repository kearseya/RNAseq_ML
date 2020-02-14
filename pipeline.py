import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV

#from sklearn.linear_model import LogisticRegression
#from sklearn.impute import SimpleImputer
#import sklearn.inspection
#from sklearn.inspection import permutation_importance #causing error
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder

from methods_functions import *


#split data into test and training data
gene_train, gene_test, gleason_train, gleason_test = train_test_split(genedata, gleasonscores, test_size=0.5)
#gene_train = genedata.copy()
#gene_test = genedata.copy()
#gleason_train = gleasonscores.copy()
#gleason_test = gleasonscores.copy()


def evaluate(model, test_features, test_labels, name):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance: ', name)
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    #print('Number of features:  ', len(model.n_features))



print("\n=====================================================\n")

#Base Random Forest Classifier
base_rfc = RandomForestClassifier(n_estimators=100, max_features=None, bootstrap=False)
base_rfc.fit(gene_train, gleason_train['Gleason'])
#evaluate(base_rfc, gene_test, gleason_test['Gleason'], "Baseline")
base_predictions = base_rfc.predict(gene_test)
print("Base RFC accuracy:   ", accuracy_score(gleason_test['Gleason'], base_predictions))
print("Base num features:   ", base_rfc.n_features_)
#print("Base num classes:    ", base_rfc.n_classes_)
#print("Base num outpurs:    ", base_rfc.n_outputs_)

print("\n=====================================================\n")

print("Random Forest Classifier")
rfc = RandomForestClassifier(n_estimators=400, max_features='sqrt', max_depth=80, min_samples_leaf=1, min_samples_split=5, bootstrap=False)
rfc_pipe = Pipeline([
   ('variance', feature_selection.VarianceThreshold()),
   ('univariance', feature_selection.SelectKBest(k=100)),
   ('from_mod', feature_selection.SelectFromModel(rfc)),
   #('PCA', PCA()),
   ('RFE', feature_selection.RFE(rfc)),
   ('rfc', rfc)])


rfc_pipe.fit(gene_train, gleason_train['Gleason'])
rfc_pipe_predictions = rfc_pipe.predict(gene_test)
print("Pipe RFC:            ", accuracy_score(gleason_test, rfc_pipe_predictions))
print("Pipe num features:   ", rfc_pipe.named_steps.rfc.n_features_)
#print("Pipe num classes:    ", pipe.named_steps.rfc.n_classes_)
#print("Pipe num outputs:    ", pipe.named_steps.rfc.n_outputs_)
#print("Pipe base estimator: ", rfc_pipe.named_steps.rfc.base_estimator_)

print("\n=====================================================\n")







"""
print("Extra Trees Classifier")
etc = ExtraTreesClassifier(n_estimators=100, bootstrap=False)
etc_pipe = Pipeline([
   ('variance', feature_selection.VarianceThreshold()),
   #('univariance', feature_selection.SelectKBest(k=100)),
   ('from_mod', feature_selection.SelectFromModel(etc)),
   #('PCA', PCA()),
   #('RFE', feature_selection.RFE(rfc)),
   ('rfc', etc)])


etc_pipe.fit(gene_train, gleason_train['Gleason'])
etc_pipe_predictions = etc_pipe.predict(gene_test)
print("Pipe ETC:            ", accuracy_score(gleason_test, etc_pipe_predictions))
print("Pipe num features:   ", etc_pipe.named_steps.rfc.n_features_)
#print("Pipe num classes:    ", pipe.named_steps.rfc.n_classes_)
#print("Pipe num outputs:    ", pipe.named_steps.rfc.n_outputs_)

print("\n=====================================================\n")

print("AdaBoostClassifier")
ada = AdaBoostClassifier(n_estimators=100)
ada_pipe = Pipeline([
   ('variance', feature_selection.VarianceThreshold()),
   #('univariance', feature_selection.SelectKBest(k=100)),
   ('from_mod', feature_selection.SelectFromModel(ada)),
   #('PCA', PCA()),
   #('RFE', feature_selection.RFE(rfc)),
   ('rfc', ada)])


ada_pipe.fit(gene_train, gleason_train['Gleason'])
ada_pipe_predictions = etc_pipe.predict(gene_test)
print("Pipe ADA:            ", accuracy_score(gleason_test, ada_pipe_predictions))
#print("Pipe num features:   ", ada_pipe.named_steps.rfc.n_features_)
#print("Pipe num classes:    ", pipe.named_steps.rfc.n_classes_)
#print("Pipe num outputs:    ", pipe.named_steps.rfc.n_outputs_)

print("\n=====================================================\n")

print("Random Forest Classifier")
rfr = RandomForestRegressor(n_estimators=400, max_features='sqrt', max_depth=80, min_samples_leaf=1, min_samples_split=5, bootstrap=False)
rfr_pipe = Pipeline([
   ('variance', feature_selection.VarianceThreshold()),
   #('univariance', feature_selection.SelectKBest(k=100)),
   ('from_mod', feature_selection.SelectFromModel(rfr)),
   #('PCA', PCA()),
   #('RFE', feature_selection.RFE(rfc)),
   ('rfc', rfc)])


rfr_pipe.fit(gene_train, gleason_train['Gleason'])
rfr_pipe_predictions = rfr_pipe.predict(gene_test)
print("Pipe RFC:            ", accuracy_score(gleason_test, rfr_pipe_predictions))
print("Pipe num features:   ", rfr_pipe.named_steps.rfc.n_features_)
#print("Pipe num classes:    ", pipe.named_steps.rfc.n_classes_)
#print("Pipe num outputs:    ", pipe.named_steps.rfc.n_outputs_)
#print("Pipe base estimator: ", rfr_pipe.named_steps.rfc.base_estimator_)

print("\n=====================================================\n")
"""






"""
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.feature_selection

X = gene_train
y = gleason_train['Gleason']
calibrated_forest = CalibratedClassifierCV(
   base_estimator=RandomForestClassifier(n_estimators=100, oob_score=True))
param_grid = {
   'base_estimator__max_depth': [2, 4, 6, 8]}
search = Gfrom sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
pipe = Pipeline([
   ('variance', feature_selection.VarianceThreshold()),
   ('select', SelectKBest()),
   #(),
   ('model', calibrated_forest)])
param_grid = {
   'select__k': [1, 2],
   'model__base_estimator__max_depth': [2, 4, 6, 8]}
print("Grid search")
search = GridSearchCV(pipe, param_grid, cv=5).fit(X, y)

#rfc = RandomForestClassifier(n_estimators=200)
#rfc.fit(gene_train, gleason_train['Gleason'])
seach_predictions = search.predict(gene_test)
print("RFC accuracy: ", accuracy_score(gleason_test['Gleason'], seach_predictions))ridSearchCV(calibrated_forest, param_grid, cv=5, error_score=np.nan)
print("Fit")
search.fit(X, y)



#pipeline 1
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2


numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessing = ColumnTransformer(
    [('num', numerical_pipe, genedata.columns.values),
    ])

rf = Pipeline([
    ('preprocess', preprocessing),
    #('ANOVA', anova_svm),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
print("Fitting first pipeline")
rf.fit(gene_train, gleason_train['Gleason'])

print("RF train accuracy: %0.3f" % rf.score(gene_train, gleason_train['Gleason']))
print("RF test accuracy: %0.3f" % rf.score(gene_test, gleason_test['Gleason']))

tree_feature_importances = (rf.named_steps['classifier'].feature_importances_)
plotimportances(tree_feature_importances, 40, "Tree")
"""

"""
#pipeline 2
grid = [{'oob_score': ['True']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
list(ParameterGrid(grid)) == [{'oob_score': 'True'},
                              {'kernel': 'rbf', 'gamma': 1},
                              {'kernel': 'rbf', 'gamma': 10}]

ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}

GridSearchCV(RandomForestClassifier())
"""

"""
removed code:

result = permutation_importance(rf, gene_test, gleason_test, n_repeats=10, random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()print("ANOVA")
anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')

anova_svm = Pipeline([
('anova', anova_filter),
('svc', clf)
])

anova_svm.fit(genedata, gleasonscores['Gleason'])
prediction = anova_svm.predict(gene_test, gleason_test['Gleason'])
print("ANOVA: ", accuracy_score(gleason_test, predictions))
print(anova_svm.score(genedata, gleasonscores['Gleason']))
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()


check_missing_val(genedata)
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

import sklearn.neural_network
from sklearn.decomposition import PCA
"""

"""
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

print("ANOVA")
anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')

anova_svm = Pipeline([
('anova', anova_filter),
('svc', clf)
])

anova_svm.fit(genedata, gleasonscores['Gleason'])
prediction = anova_svm.predict(gene_test, gleason_test['Gleason'])
print("ANOVA: ", accuracy_score(gleason_test, predictions))
print(anova_svm.score(genedata, gleasonscores['Gleason']))
"""


"""
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)

estimators = [('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=42)),
              ('RANSAC', RANSACRegressor(random_state=42)),
              ('HuberRegressor', HuberRegressor())]
colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen', 'HuberRegressor': 'black'}
linestyle = {'OLS': '-', 'Theil-Sen': '-.', 'RANSAC': '--', 'HuberRegressor': '--'}
lw = 3

X_plot = np.linspace(genedata.min(), genedata.max())
for title, this_X, this_y in [
        ('Modeling Errors Only', genedata, gleasonscores['Gleason']),
        ('Corrupt X, Small Deviants', genedata.std(), gleasonscores['Gleason']),
        ('Corrupt y, Small Deviants', genedata, gleasonscores['Gleason'].std()),
        ('Corrupt X, Large Deviants', genedata.std()*2, gleasonscores['Gleason']),
        ('Corrupt y, Large Deviants', genedata, gleasonscores['Gleason'].std()*2)]:
    plt.figure(figsize=(5, 4))
    plt.plot(this_X[:, 0], this_y, 'b+')

    for name, estimator in estimators:
        model = make_pipeline(PolynomialFeatures(3), estimator)
        model.fit(this_X, this_y)
        mse = mean_squared_error(model.predict(gene_test), gleason_test['Gleason'])
        y_plot = model.predict(x_plot[:, np.newaxis])
        plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
                 linewidth=lw, label='%s: error = %.3f' % (name, mse))

    legend_title = 'Error of Mean\nAbsolute Deviation\nto Non-corrupt Data'
    legend = plt.legend(loc='upper right', frameon=False, title=legend_title,
                        prop=dict(size='x-small'))
    plt.xlim(-4, 10.2)
    plt.ylim(-2, 10.2)
    plt.title(title)
plt.show()
"""
