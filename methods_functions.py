import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import time

import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn import feature_selection
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

#import sklearn.neural_network
#from sklearn.linear_model import LogisticRegression
#from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectKBest, RFE



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

#only one person with gleason score of 5, causes error
#print(gleasonscores['Gleason'].value_counts())
gleasonscores = gleasonscores.drop("PT264")
genedata = genedata.drop("PT264")
#only 5 people with a score of 9, here for experimentation
#genedata = genedata.drop(["CM.4-0065", "CM.1-0001.1", "CM.1-0015.1", "CM.1-0018.1", "CMO10311-0007"])
#gleasonscores = gleasonscores.drop(["CM.4-0065", "CM.1-0001.1", "CM.1-0015.1", "CM.1-0018.1", "CMO10311-0007"])


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
def plotimportances(fitdata, gene_data, num_features, method):
    feature_importance = fitdata
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    sorted_idx = sorted_idx[-int(num_features):-1:1]
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, gene_data.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance: '+method, fontsize=30)
    plt.tick_params(axis='x', which='major', labelsize=15)
    sns.despine(left=True, bottom=True)
    plt.show()

#plotimportances(initial_feature_importance, gene_data, 40, "Initial")



def variance_filter(gene_data, gleason_score):
    print("\n=====================================================\n")
    method = "Variance filter"
    #remove features with low variance
    print("Removing features with low variance")
    t1 = time.time()
    sel = feature_selection.VarianceThreshold()
    print("Fitting and Transforming")
    train_variance = sel.fit_transform(gene_data)
    passed_var_filter = sel.get_support()
    before_num_genes = gene_data.shape[1]

    print("Before:  ", str(before_num_genes))
    low_var_filter_genes = []
    for bool, gene in zip(passed_var_filter, all_gene_names):
        if bool:
            low_var_filter_genes.append(gene)

    gene_data = pd.DataFrame(train_variance, columns=low_var_filter_genes)
    #gene_data = gene_data.set_index(row_names)
    print("After:   ", str(gene_data.shape[1]), str(np.bool(gene_data.shape[1]==train_variance.shape[1])))
    print("Removed: ", str(before_num_genes-gene_data.shape[1]))

    rfc = RandomForestClassifier(n_estimators=100)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, verbose=True, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    print("Plotting importances")
    rfc.fit(gene_data, gleason_score['Gleason'])
    rfc_importances = rfc.feature_importances_
    plotimportances(rfc_importances, gene_data, 50, method)

    global methodlog
    methodlog.append({"method": method, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = variance_filter(gene_data, gleason_score)




def univariate_filter(gene_data, gleason_score):
    print("\n=====================================================\n")
    method = "Univariate filter"
    #univariate feature selection
    #stats tests to select features with highest correlation to target
    print("Univariate feature selection")
    #feature extraction
    print("Feature selection")
    t1 = time.time()
    k_best = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=5000)
    #fit on training data and transform
    print("Fitting and Transforming")
    univariate_features = k_best.fit_transform(gene_data, gleason_score['Gleason'])

    gene_names = list(gene_data.columns.values)
    before_num_genes = gene_data.shape[1]
    passed_univariate_filter = k_best.get_support()
    print("Before:  ", str(before_num_genes))
    univariate_filter_genes = []
    for bool, gene in zip(passed_univariate_filter, gene_names):
        if bool:
            univariate_filter_genes.append(gene)

    gene_data = pd.DataFrame(univariate_features, columns=univariate_filter_genes)
    #gene_data = gene_data.set_index(row_names)
    print("After:   ", str(gene_data.shape[1]), str(np.bool(gene_data.shape[1]==univariate_features.shape[1])))
    print("Removed: ", str(before_num_genes-gene_data.shape[1]))

    rfc = RandomForestClassifier(n_estimators=100)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, verbose=True, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    #checking which are the most important features
    print("Plotting importance")
    k_feature_importance = rfc.fit(gene_data, gleason_score['Gleason']).feature_importances_
    plotimportances(k_feature_importance, gene_data, 50, method)

    global methodlog
    methodlog.append({"method": method, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = univariate_filter(gene_data, gleason_score)



def correlation_filter(gene_data, gleason_score):
    print("\n=====================================================\n")
    method = "Correlation filter"
    #remove highly correlated features
    #this prevents overfitting (due to highly correlated or colinear features)
    print("Calculating correlation matrix (this may take some time, can crash n>1000)")
    t1 = time.time()
    corr_matrix = gene_data.corr().abs()
    ##print(corr_matrix['Gleason'].sort_values(ascending=False).head(10))
    #plot correlation matrix
    matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    sns.heatmap(matrix)
    plt.show()

    #find index of feature columns with high correlation
    to_drop = [column for column in matrix.columns if any(matrix[column] > 0.50)]
    if "Gleason" in to_drop:
        to_drop.remove("Gleason")
    print("Before:  ", str(gene_data.shape[1]))
    gene_data = gene_data.drop(to_drop, axis=1)
    print("After:   ", str(gene_data.shape[1]))
    print("Removed: ", str(len(to_drop)))

    rfc = RandomForestClassifier(n_estimators=100)
    #cross validating
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, verbose=True, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    #checking which are the most important features
    print("Plotting importance")
    correlation_importance = rfc.fit(gene_data, gleason_score['Gleason']).feature_importances_
    plotimportances(correlation_importance, gene_data, 50, method)

    global methodlog
    methodlog.append({"method": method, "size": gene_data.shape[1], "removed": len(to_drop), "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = correlation_filter(gene_data, gleason_score)



def recursive_feature_elimination(gene_data, gleason_score):
    print("\n=====================================================\n")
    method = "Recursive feature elimination"
    #recursive feature elimination
    print("Recursive feature elimination")
    t1 = time.time()
    #lr = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=5000) #max_iter specified as does not converge at 1000 (default)
    rfc = RandomForestClassifier(n_estimators=100)
    #feature extraction
    print("Feature extraction")
    rfe = feature_selection.RFE(rfc, verbose=True)
    #fit on train set and transform
    print("Fitting and Transforming")
    RFE_features = rfe.fit_transform(gene_data, gleason_score['Gleason'])

    gene_names = list(gene_data.columns.values)
    before_num_genes = gene_data.shape[1]
    passed_RFE_filter = rfe.get_support()
    print("Before:  ", str(before_num_genes))
    RFE_filter_genes = []
    for bool, gene in zip(passed_RFE_filter, gene_names):
        if bool:
            RFE_filter_genes.append(gene)

    gene_data = pd.DataFrame(RFE_features, columns=RFE_filter_genes)
    #gene_data = gene_data.set_index(row_names)
    print("After:   ", str(gene_data.shape[1]), str(np.bool(gene_data.shape[1]==RFE_features.shape[1])))
    print("Removed: ", str(before_num_genes-gene_data.shape[1]))

    rfc = RandomForestClassifier(n_estimators=100)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    print("Plotting importance")
    recursive_importance = rfc.fit(gene_data, gleason_score['Gleason']).feature_importances_
    plotimportances(recursive_importance, gene_data, 50, method)

    global methodlogp
    methodlog.append({"method": method, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = recursive_feature_elimination(gene_data, gleason_score)



def feature_select_from_model(gene_data, gleason_score):
    print("\n=====================================================\n")
    method = "Feature select \nfrom model"
    t1 = time.time()
    rfc = RandomForestClassifier(n_estimators=100)
    #feature selection from model
    #feature extraction
    print("Feature selection from model")
    select_model = feature_selection.SelectFromModel(rfc)
    #fit on train set
    print("Fitting")
    sel_fit = select_model.fit(gene_data, gleason_score['Gleason'])
    #transform train set
    print("Transforming")
    model_features = sel_fit.transform(gene_data)

    gene_names = list(gene_data.columns.values)
    before_num_genes = gene_data.shape[1]
    passed_sel_filter = select_model.get_support()
    print("Before:  ", str(before_num_genes))
    sel_filter_genes = []
    for bool, gene in zip(passed_sel_filter, gene_names):
        if bool:
            sel_filter_genes.append(gene)

    gene_data = pd.DataFrame(model_features, columns=sel_filter_genes)
    #gene_data = gene_data.set_index(row_names)
    print("After:   ", str(gene_data.shape[1]), str(np.bool(gene_data.shape[1]==model_features.shape[1])))
    print("Removed: ", str(before_num_genes-gene_data.shape[1]))

    rfc = RandomForestClassifier(n_estimators=100)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    print("Plotting importance")
    featsel_importances = rfc.fit(gene_data, gleason_score['Gleason']).feature_importances_
    #plotimportances(featsel_importances, gene_data, 50, method)

    global methodlog
    methodlog.append({"method": method, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = feature_select_from_model(gene_data, gleason_score)



def tree_based_selection(gene_data, gleason_score):
    print("\n=====================================================\n")
    method = "Tree based selection"
    #tree based selection
    print(method)
    t1 = time.time()
    etc = ExtraTreesClassifier(n_estimators=100)
    print("Fitting")
    etc = etc.fit(gene_data, gleason_score['Gleason'])
    etc_model = feature_selection.SelectFromModel(etc, prefit=True)
    print("Transforming")
    etc_features = etc_model.transform(gene_data)


    gene_names = list(gene_data.columns.values)
    before_num_genes = gene_data.shape[1]
    passed_etc_filter = etc_model.get_support()
    print("Before:  ", str(before_num_genes))
    etc_filter_genes = []
    for bool, gene in zip(passed_etc_filter, gene_names):
        if bool:
            etc_filter_genes.append(gene)

    gene_data = pd.DataFrame(etc_features, columns=etc_filter_genes)
    #gene_data = gene_data.set_index(row_names)
    print("After:   ", str(gene_data.shape[1]), str(np.bool(gene_data.shape[1]==etc_features.shape[1])))
    print("Removed: ", str(before_num_genes-gene_data.shape[1]))

    rfc = RandomForestClassifier(n_estimators=100)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))
    print("Plotting importances")
    etc_importances = etc.feature_importances_
    #plotimportances(etc_importances, gene_data, 2, method)

    global methodlog
    methodlog.append({"method": method, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = tree_based_selection(gene_data, gleason_score, mergedset)


def L1_based_select(gene_data, gleason_score):
    print("\n=====================================================\n")
    method = "L1-based selection"
    print(method)
    #lr = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=5000)
    print("Fitting")
    t1 = time.time()
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(gene_data, gleason_score['Gleason'])
    L1_model = feature_selection.SelectFromModel(lsvc, prefit=True)
    #L1_importances = lsvc.feature_importances_ #not supported with current method
    #plotimportances(L1_importances, gene_data, len(L1_importances), method)
    print("Transforming")
    model_features = L1_model.transform(gene_data)

    gene_names = list(gene_data.columns.values)
    before_num_genes = gene_data.shape[1]
    passed_L1_filter = L1_model.get_support()
    print("Before:  ", str(before_num_genes))
    L1_filter_genes = []
    for bool, gene in zip(passed_L1_filter, gene_names):
        if bool:
            L1_filter_genes.append(gene)

    gene_data = pd.DataFrame(model_features, columns=L1_filter_genes)
    #gene_data = gene_data.set_index(row_names)
    print("After:   ", str(gene_data.shape[1]), str(np.bool(gene_data.shape[1]==model_features.shape[1])))
    print("Removed: ", str(before_num_genes-gene_data.shape[1]))

    rfc = RandomForestClassifier(n_estimators=100)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    print("Plotting importances")
    L1_importance = rfc.fit(gene_data, gleason_score['Gleason']).feature_importances_
    #plotimportances(L1_importance, gene_data, 2, method)

    global methodlog
    methodlog.append({"method": method, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = L1_based_select(gene_data, gleason_score)


#still not sure, needs fixing, works in pipeline
def PCA_analysis(gene_data, gleason_score, n_comp):
    print("\n=====================================================\n")
    method = "PCA analysis"
    # pca - keep 90% of variance
    print("Prinipal component analysis")
    t1 = time.time()
    pca = PCA(0.95, n_components=n_comp, svd_solver='full')

    print("Fitting and Transforming")
    principal_components = pca.fit_transform(gene_data)
    principal_df = pd.DataFrame(data = principal_components)
    print(principal_df.shape)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    rfc = RandomForestClassifier(n_estimators=100)

    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(something.columns.values)], gleasonscores['Gleason'], cv=5)

    print("RFC Scores:  ", rfc_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    global methodlog
    methodlog.append({"method": method, "size": gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    #return gene_data, gleason_score

#PCA_analysis(gene_data, gleason_score)



#print(methodlog)

def visualise_accuracy_methodlog():
    methodorder = []
    size = []
    rfc_accuracy = []
    rfc_error = []
    for i in range(len(methodlog)):
        methodorder.append(methodlog[i]['method'])
        size.append(methodlog[i]['size'])
        try:
            rfc_accuracy.append(np.mean(methodlog[i]['rfc_scores']))
            rfc_error.append(np.std(methodlog[i]['rfc_scores'])*2)
        except:
            rfc_accuracy.append(0.0)
            rfc_error.append(0.0)

    ind = np.arange(len(methodlog))
    width = 0.45
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, tuple(rfc_accuracy), width, color='royalblue', yerr=tuple(rfc_error))
    #rects2 = ax.bar(ind+width, tuple(rfr_accuracy), width, color='seagreen', yerr=tuple(rfr_error))
    # add some
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross validation scores\n for RFC')
    ax.set_xticks(ind) #+ width / 2)
    ax.set_xticklabels(tuple(methodorder))

    #ax.legend( (rects1[0]), ('Random Forest Classifier') ) #, rects2[0]), ('Logistic Regression', 'Random Forest Classifier') )

    plt.show()











"""
code removed:
#Don't think logistic regression will work (not binary)
#lr_scores = cross_val_score(lr, univariate_features, gleason_score['Gleason'], cv=5, verbose=True)
#print("LR Scores:   ", lr_scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (lr_scores.mean(), lr_scores.std() * 2))

#print(k_fit)
#transform training data
print("Transforming")
univariate_features = k_fit.transform(mergedset)
print(univariate_features.shape)

#transform train set
#print("Transforming")
#recursive_features = RFE_features.transform(mergedset)

#DOESN'T WORK, CHECK ATTRIBUTES
#mod_sel_importances = sel_fit.feature_importances_
#plotimportances(mod_sel_importances, gene_data, len(mod_sel_importances), method)

#DOESN'T WORK, CHECK ATTRIBUTES
#PCA_importances = pca.fit(principal_components, gleason_score['Gleason']).feature_importances_
#plotimportances(PCA_importances, gene_data, len(PCA_importances), method)

lr = LogisticRegression(solver='liblinear', multi_class='auto')
rfc = RandomForestClassifier(n_estimators=100)
print("Cross validating")
lr_scores = cross_val_score(lr, principal_df, gleason_score['Gleason'], cv=5)
rfc_scores = cross_val_score(rfc, principal_df, gleason_score['Gleason'], cv=5)
print("LR  Scores:   ", lr_scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (lr_scores.mean(), lr_scores.std() * 2))
print("RFC Scores:  ", rfc_scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))
gene_names = list(gene_data.columns.values)
before_num_genes = gene_data.shape[1]
passed_pca_filter = False
print("Before:  ", str(before_num_genes))
pca_filter_genes = []
for bool, gene in zip(passed_pca_filter, gene_names):
    if bool:
        pca_filter_genes.append(gene)
gene_data = pd.DataFrame(principal_components, columns=pca_filter_genes)
#gene_data = gene_data.set_index(row_names)
print("After:   ", str(gene_data.shape[1]), str(np.bool(gene_data.shape[1]==principal_components.shape[1])))
print("Removed: ", str(before_num_genes-gene_data.shape[1]))
global methodlog
methodlog.append({"method": method, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "lr_scores": lr_scores, "rfc_scores": rfc_scores, "lr_acc": lr_scores.mean(), "rfc_acc": rfc_scores.mean(), "lr_err": lr_scores.std()*2, "rfc_err": rfc_scores.std()*2})
"""
