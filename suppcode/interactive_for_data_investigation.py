import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from pprint import pprint

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn import feature_selection
from sklearn.model_selection import train_test_split

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

print(gleasonscores['Gleason'].value_counts())

print(gleasonscores[gleasonscores.Gleason == 5])
print(gleasonscores[gleasonscores.Gleason == 9])

print(gleasonscores['Gleason'])
print(gleasonscores.values.ravel())


"""
#split data into test and training data
X_train, X_test, y_train, y_test = train_test_split(genedata, gleasonscores, test_size=0.05)



sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
print("Fit")
sel.fit(X_train, y_train['Gleason'])

print("Selected features")
selected_feat= X_train.columns[(sel.get_support())]
len(selected_feat)

print(selected_feat)

plt.plot(list(sel.estimator_.feature_importances_))
plt.show()
"""
