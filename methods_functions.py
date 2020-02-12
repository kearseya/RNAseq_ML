import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import sklearn.neural_network
from sklearn.decomposition import PCA


#read in the data
data_file_name = "GSE54460_FPKM-genes-TopHat2-106samples-12-4-13.txt"

rawdata = pd.read_table(data_file_name, sep='\t', index_col=1, low_memory=False)
#rawdata = rawdata.rename(columns={"Unnamed: 1": "GENE"})

#seperate gene data
genedata = rawdata[15:].T[1:]
print("Gene data")
print(genedata)
#seperate gleason score data
gleasonscores = rawdata.loc[rawdata['Unnamed: 0'] == "tgleason"].T[1:]
gleasonscores = gleasonscores.rename(columns={np.nan: "Gleason"})
print("Gleason scores")
print(gleasonscores)

#remove strings from tables
genedata = genedata.apply(pd.to_numeric, errors='coerce').fillna(0.000001)
gleasonscores = gleasonscores.apply(pd.to_numeric, errors='coerce').fillna(0.000001)

#create a combined matrix of genes and gleason scores
mergedset = pd.merge(genedata, gleasonscores, left_index=True, right_index=True)

#create score log
methodlog = []

#make a copy of original gene data
ogenedata = genedata.copy()
#create a list of rownames
row_names = rawdata.columns.values[1:]
#create a list of strating gene gene names
all_gene_names = list(genedata.columns.values)


def check_missing_val(genedata):
    #check for missing values
    print("Checking for missing data: ")
    print(genedata.isnull().any().any())
    #FALSE so there are no missing values


def initial_rank(genedata, gleasonscores):
    #create RFC and fit the data to it
    rfc = RandomForestClassifier(n_estimators=100)
    print("Fitting model to random forest classifier")
    rfc1 = rfc.fit(genedata, gleasonscores['Gleason'])

    #rank the importance of the features
    print("Ranking features")
    initial_feature_importance = rfc1.feature_importances_
    return initial_feature_importance


#produce graph showing the relatvie importance of the top features
def plotimportances(fitdata, num_features, method):
    feature_importance = fitdata
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    sorted_idx = sorted_idx[-int(num_features):-1:1]
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, genedata.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance: '+method, fontsize=30)
    plt.tick_params(axis='x', which='major', labelsize=15)
    sns.despine(left=True, bottom=True)
    plt.show()

#plotimportances(initial_feature_importance, 40, "Initial")



def variance_filter(genedata, gleasonscores):
    print("\n=====================================================\n")
    method = "Variance filter"
    #remove features with low variance
    print("Removing features with low variance")
    sel = feature_selection.VarianceThreshold()
    train_variance = sel.fit_transform(genedata)
    passed_var_filter = sel.get_support()
    before_num_genes = genedata.shape[1]

    print("Before:  ", str(before_num_genes))
    low_var_filter_genes = []
    for bool, gene in zip(passed_var_filter, all_gene_names):
        if bool:
            low_var_filter_genes.append(gene)

    genedata = pd.DataFrame(train_variance, columns=low_var_filter_genes)
    genedata = genedata.set_index(row_names)
    print("After:   ", str(genedata.shape[1]), str(np.bool(genedata.shape[1]==train_variance.shape[1])))
    print("Removed: ", str(before_num_genes-genedata.shape[1]))

    global methodlog
    methodlog.append({"method": method, "size": genedata.shape[1]})

    #remake merged dataset
    mergedset = pd.merge(genedata, gleasonscores, left_index=True, right_index=True)

    print("\n=====================================================\n")
    return genedata, gleasonscores, mergedset

#genedata, gleasonscores, mergedset = variance_filter(genedata, gleasonscores)




def univariate_filter(genedata, gleasonscores):
    print("\n=====================================================\n")
    method = "Univariate filter"
    #univariate feature selection
    #stats tests to select features with highest correlation to target
    print("Univariate feature selection")
    #feature extraction
    print("Feature selection")
    k_best = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=500)
    #fit on training data and transform
    print("Fitting and Transforming")
    univariate_features = k_best.fit_transform(genedata, gleasonscores['Gleason'])

    lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=5000)
    rfc = RandomForestClassifier(n_estimators=2500)

    print("Cross validating")
    lr_scores = cross_val_score(lr, univariate_features, gleasonscores['Gleason'], cv=5, verbose=True)
    rfc_scores = cross_val_score(rfc, univariate_features, gleasonscores['Gleason'], cv=5, verbose=True)

    print("LR Scores:   ", lr_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (lr_scores.mean(), lr_scores.std() * 2))
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    #checking which are the most important features
    print("Feature selection from K")
    k_feature_importance = rfc.fit(univariate_features, gleasonscores['Gleason']).feature_importances_

    print("Plotting univariate features ")
    plotimportances(k_feature_importance, 50, method)

    gene_names = list(genedata.columns.values)
    before_num_genes = genedata.shape[1]
    passed_univariate_filter = k_best.get_support()
    print("Before:  ", str(before_num_genes))
    univariate_filter_genes = []
    for bool, gene in zip(passed_univariate_filter, gene_names):
        if bool:
            univariate_filter_genes.append(gene)

    genedata = pd.DataFrame(univariate_features, columns=univariate_filter_genes)
    genedata = genedata.set_index(row_names)
    print("After:   ", str(genedata.shape[1]), str(np.bool(genedata.shape[1]==univariate_features.shape[1])))
    print("Removed: ", str(before_num_genes-genedata.shape[1]))

    global methodlog
    methodlog.append({"method": method, "size": genedata.shape[1], "lr_scores": lr_scores, "rfc_scores": rfc_scores, "lr_acc": lr_scores.mean(), "rfc_acc": rfc_scores.mean(), "lr_err": lr_scores.std()*2, "rfc_err": rfc_scores.std()*2})

    #remake merged dataset
    mergedset = pd.merge(genedata, gleasonscores, left_index=True, right_index=True)

    print("\n=====================================================\n")
    return genedata, gleasonscores, mergedset

#genedata, gleasonscores, mergedset = univariate_filter(genedata, gleasonscores)



def correlation_filter(genedata, gleasonscores, mergedset):
    print("\n=====================================================\n")
    method = "Correlation filter"
    #remove highly correlated features
    #this prevents overfitting (due to highly correlated or colinear features)
    print("Calculating correlation matrix (this may take some time)")
    corr_matrix = mergedset.corr().abs()
    print(corr_matrix['Gleason'].sort_values(ascending=False).head(10))
    #plot correlation matrix
    matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    sns.heatmap(matrix)
    plt.show()

    #find index of feature columns with high correlation
    to_drop = [column for column in matrix.columns if any(matrix[column] > 0.50)]
    if "Gleason" in to_drop:
        to_drop.remove("Gleason")
    print("Before:  ", str(genedata.shape[1]))
    genedata = genedata.drop(to_drop, axis=1)
    print("After:   ", str(genedata.shape[1]))
    print("Removed: ", str(len(to_drop)))

    global methodlog
    methodlog.append({"method": method, "size": genedata.shape[1]})

    #remake merged dataset
    mergedset = pd.merge(genedata, gleasonscores, left_index=True, right_index=True)

    print("\n=====================================================\n")
    return genedata, gleasonscores, mergedset

#genedata, gleasonscores, mergedset = correlation_filter(genedata, gleasonscores, mergedset)



def recursive_feature_elimination(genedata, gleasonscores):
    print("\n=====================================================\n")
    method = "Recursive feature elimination"
    #recursive feature elimination
    print("Recursive feature elimination")
    lr = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=5000) #max_iter specified as does not converge at 1000 (default)

    #feature extraction
    print("Feature extraction")
    rfe = feature_selection.RFE(lr, verbose=True)
    #fit on train set and transform
    print("Fitting and Transforming")
    RFE_features = rfe.fit_transform(genedata, gleasonscores['Gleason'])

    lr = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=5000)
    rfc = RandomForestClassifier(n_estimators=2500)

    print("Cross validating")
    lr_scores = cross_val_score(lr, RFE_features, gleasonscores['Gleason'], cv=5)
    rfc_scores = cross_val_score(rfc, RFE_features, gleasonscores['Gleason'], cv=5)

    print("LR  Scores:   ", lr_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (lr_scores.mean(), lr_scores.std() * 2))
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    rfe_importances = rfc.fit(RFE_features, gleasonscores['Gleason']).feature_importances_
    plotimportances(rfe_importances, len(rfe_importances), method)

    gene_names = list(genedata.columns.values)
    before_num_genes = genedata.shape[1]
    passed_RFE_filter = rfe.get_support()
    print("Before:  ", str(before_num_genes))
    RFE_filter_genes = []
    for bool, gene in zip(passed_RFE_filter, gene_names):
        if bool:
            RFE_filter_genes.append(gene)

    genedata = pd.DataFrame(RFE_features, columns=RFE_filter_genes)
    genedata = genedata.set_index(row_names)
    print("After:   ", str(genedata.shape[1]), str(np.bool(genedata.shape[1]==RFE_features.shape[1])))
    print("Removed: ", str(before_num_genes-genedata.shape[1]))

    global methodlog
    methodlog.append({"method": method, "size": genedata.shape[1], "lr_scores": lr_scores, "rfc_scores": rfc_scores, "lr_acc": lr_scores.mean(), "rfc_acc": rfc_scores.mean(), "lr_err": lr_scores.std()*2, "rfc_err": rfc_scores.std()*2})

    #remake merged dataset
    mergedset = pd.merge(genedata, gleasonscores, left_index=True, right_index=True)

    print("\n=====================================================\n")
    return genedata, gleasonscores, mergedset

#genedata, gleasonscores, mergedset = recursive_feature_elimination(genedata, gleasonscores)



def feature_select_from_model(genedata, gleasonscores):
    print("\n=====================================================\n")
    method = "Feature select \nfrom model"
    lr = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=5000)
    #feature selection from model
    #feature extraction
    print("Feature selection from model")
    select_model = feature_selection.SelectFromModel(lr)
    #fit on train set
    print("Fitting")
    sel_fit = select_model.fit(genedata, gleasonscores['Gleason'])
    #transform train set
    print("Transforming")
    model_features = sel_fit.transform(genedata)

    lr = LogisticRegression(solver='liblinear', multi_class='auto')
    rfc = RandomForestClassifier(n_estimators=500)

    lr_scores = cross_val_score(lr, model_features, gleasonscores['Gleason'], cv=5)
    rfc_scores = cross_val_score(rfc, model_features, gleasonscores['Gleason'], cv=5)

    print("LR  Scores:   ", lr_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (lr_scores.mean(), lr_scores.std() * 2))
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    global methodlog
    methodlog.append({"method": method, "size": genedata.shape[1], "lr_scores": lr_scores, "rfc_scores": rfc_scores, "lr_acc": lr_scores.mean(), "rfc_acc": rfc_scores.mean(), "lr_err": lr_scores.std()*2, "rfc_err": rfc_scores.std()*2})

    #remake merged dataset
    mergedset = pd.merge(genedata, gleasonscores, left_index=True, right_index=True)

    print("\n=====================================================\n")
    return genedata, gleasonscores, mergedset

#genedata, gleasonscores, mergedset = feature_select_from_model(genedata, gleasonscores)



def PCA_filter(genedata, gleasonscores):
    print("\n=====================================================\n")
    method = "PCA filter"
    # pca - keep 90% of variance
    print("Prinipal components filter")
    pca = PCA(0.90)

    print("Fitting and Transforming")
    principal_components = pca.fit_transform(genedata)
    principal_df = pd.DataFrame(data = principal_components)
    principal_df.shape

    lr = LogisticRegression(solver='liblinear')
    rfc = RandomForestClassifier(n_estimators=100)

    print("Cross validating")
    lr_scores = cross_val_score(lr, principal_df, gleasonscores['Gleason'], cv=5)
    rfc_scores = cross_val_score(rfc, principal_df, gleasonscores['Gleason'], cv=5)

    print("LR  Scores:   ", lr_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (lr_scores.mean(), lr_scores.std() * 2))
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    global methodlog
    methodlog.append({"method": method, "size": genedata.shape[1], "lr_scores": lr_scores, "rfc_scores": rfc_scores, "lr_acc": lr_scores.mean(), "rfc_acc": rfc_scores.mean(), "lr_err": lr_scores.std()*2, "rfc_err": rfc_scores.std()*2})

    #remake merged dataset
    mergedset = pd.merge(genedata, gleasonscores, left_index=True, right_index=True)

    print("\n=====================================================\n")
    return genedata, gleasonscores, mergedset

#genedata, gleasonscores, mergedset = PCA_filter(genedata, gleasonscores)



#print(methodlog)


def visualise_accuracy_methodlog():
    methodorder = []
    size = []
    lr_accuracy = []
    rfc_accuracy = []
    lr_error = []
    rfc_error = []
    for i in range(len(methodlog)):
        methodorder.append(methodlog[i]['method'])
        size.append(methodlog[i]['size'])
        try:
            lr_accuracy.append(methodlog[i]['lr_acc'])
            rfc_accuracy.append(methodlog[i]['rfc_acc'])
            lr_error.append(methodlog[i]['lr_err'])
            rfc_error.append(methodlog[i]['rfc_err'])
        except:
            lr_accuracy.append(0.0)
            rfc_accuracy.append(0.0)
            lr_error.append(0.0)
            rfc_error.append(0.0)

    ind = np.arange(len(methodlog))
    width = 0.35
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, tuple(lr_accuracy), width, color='royalblue', yerr=tuple(lr_error))
    rects2 = ax.bar(ind+width, tuple(rfc_accuracy), width, color='seagreen', yerr=tuple(rfc_error))
    # add some
    ax.set_ylabel('Accuracy')
    ax.set_title('Scores for LR and RFC')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(tuple(methodorder))

    ax.legend( (rects1[0], rects2[0]), ('Logistic Regression', 'Random Forest Classifier') )

    plt.show()











"""
code removed:

#print(k_fit)
#transform training data
print("Transforming")
univariate_features = k_fit.transform(mergedset)
print(univariate_features.shape)

# transform train set
#print("Transforming")
#recursive_features = RFE_features.transform(mergedset)

#DOESN'T WORK
#mod_sel_importances = sel_fit.feature_importances_
#plotimportances(mod_sel_importances, len(mod_sel_importances), method)

#DOESN'T WORK
#PCA_importances = pca.fit(principal_components, gleasonscores['Gleason']).feature_importances_
#plotimportances(PCA_importances, len(PCA_importances), method)
"""
