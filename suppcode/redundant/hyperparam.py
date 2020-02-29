import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import feature_selection

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


#read in the data
data_file_name = "GSE54460_FPKM-genes-TopHat2-106samples-12-4-13.txt"

rawdata = pd.read_table(data_file_name, sep='\t', index_col=1, low_memory=False)
#rawdata = rawdata.rename(columns={"Unnamed: 1": "GENE"})

#seperate gene data
genedata = rawdata[15:].T[1:]
#print("Gene data")
#print(genedata)
#seperate gleason score data
gleasonscores = rawdata.loc[rawdata['Unnamed: 0'] == "tgleason"].T[1:]
gleasonscores = gleasonscores.rename(columns={np.nan: "Gleason"})
#print("Gleason scores")
#print(gleasonscores)

#remove strings from tables
genedata = genedata.apply(pd.to_numeric, errors='coerce').fillna(0.0)
gleasonscores = gleasonscores.apply(pd.to_numeric, errors='coerce').fillna(0.0)

gleasonscores = gleasonscores.drop("PT264")
genedata = genedata.drop("PT264")

#split data into test and training data
gene_train, gene_test, gleason_train, gleason_test = train_test_split(genedata, gleasonscores, test_size=0.5)

#genedata = genedata[features_selected]
#train_features = gene_data[features_selected].copy()
#test_features = genedata[features_selected].copy()
#train_labels = gleasonscores['Gleason'].copy()
#test_labels = gleasonscores['Gleason'].copy()

from test_script import gene_train_filt

train_features = gene_train[list(gene_train_filt.columns.values)].copy()
test_features = gene_test[list(gene_train_filt.columns.values)].copy()
train_labels = gleason_train['Gleason']
test_labels = gleason_test['Gleason']



from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#pprint(random_grid)



rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features, train_labels)

print("Best parameters")
pprint(rf_random.best_params_)



def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy
base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(train_features, train_labels)
grid_search.best_params_

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_features, test_labels)

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


final_model = grid_search.best_estimator_

print('Final Model Parameters:\n')
pprint(final_model.get_params())
print('\n')
grid_final_accuracy = evaluate(final_model, test_features, test_labels)
evaluate(final_model, test_features, test_labels)

"""
comparison = {'model': [baseline_results['model'], one_year_results['model']],
              'accuracy': [round(baseline_results['accuracy'], 3), round(one_year_results['accuracy'], 3)],
              'error': [round(baseline_results['error'], 3), round(one_year_results['error'], 3)],
              'n_features': [baseline_results['n_features'], one_year_results['n_features']],
              'n_trees': [baseline_results['n_trees'], int(one_year_results['n_trees'])],
              'time': [round(baseline_results['time'], 4), round(one_year_results['time'], 4)]}

for model in [four_year_results, four_years_important_results, random_results, first_grid_results, final_model_results]:
    comparison['accuracy'].append(round(model['accuracy'], 3))
    comparison['error'].append(round(model['error'], 3))
    comparison['model'].append(model['model'])
    comparison['n_features'].append(model['n_features'])
    comparison['n_trees'].append(int(model['n_trees']))
    comparison['time'].append(round(model['time'], 4))

xvalues = list(range(len(comparison)))
plt.subplots(1, 2, figsize=(10, 6))
plt.subplot(121)
plt.bar(xvalues, comparison['accuracy'], color = 'g', edgecolor = 'k', linewidth = 1.8)
plt.xticks(xvalues, comparison['model'], rotation = 45, fontsize = 12)
plt.ylim(ymin = 91, ymax = 94)
plt.xlabel('model'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy Comparison');

plt.subplot(122)
plt.bar(xvalues, comparison['error'], color = 'r', edgecolor = 'k', linewidth = 1.8)
plt.xticks(xvalues, comparison['model'], rotation = 45)
plt.ylim(ymin = 3.5, ymax = 4.8)
plt.xlabel('model'); plt.ylabel('Error (deg)'); plt.title('Error Comparison');
plt.show();
"""



"""
#GRIDSEARCH METHOD
n_range = range(10, 250, 10)
param_grid = dict(n_estimators = n_range)

rfc = RandomForestClassifier()

grid = GridSearchCV(rfc, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

#print(grid.cv_results_)
#grid_mean_scores = [result.mean_test_score for result in grid.cv_results_]
plt.plot(n_range, grid.cv_results_["mean_test_score"])
plt.xlabel("Value for N in RFC")
plt.ylabel("Cross-Validation Accuracy")
plt.show()
"""
