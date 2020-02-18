import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import time
import itertools

import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn import feature_selection
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate, cross_val_score, learning_curve,validation_curve
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

from sklearn.tree import export_graphviz
from subprocess import call
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix

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
#sample bias to Gleason score 7 (80/106), lead to underfitting, removed to prevent bias
genedata = genedata.drop(['CM.4-0014', 'CM.4-0028', 'CM.4-0054', 'CM.4-0055', 'CM.4-0074',
       'CM.4-0078', 'CM.4-0080', 'CM.4-0082', 'CM.4-0093', 'CM.1-0028.1',
       'CM.4-0019', 'CM.4-0064', 'CM.4-0079', 'CM.4-0047', 'CM.4-0048',
       'CM.4-0050', 'CM.4-0051', 'CM.4-0052', 'CM.4-0061', 'CM.4-0076',
       'CM.4-0083', 'CM.4-0087', 'CM.4-0094', 'CM.4-0013', 'CM.4-0067',
       'CM.4-0068', 'CM.4-0070', 'CM.4-0090', 'CM.4-0098', 'CM.4-0045',
       'CM.4-0075', 'CM.4-0081', 'CM.4-0091', 'PT184', 'PT199', 'PT236',
       'PT243', 'CM.4-0066', 'CM.4-0095', 'PT081', 'CMO10311-0036',
       'CM.1-0003.1', 'CM.1-0004.1', 'CM.1-0017.1', 'CM.1-0025.1', 'CM.4-0042',
       'CM.4-0043', 'CM.4-0044', 'CM.4-0046', 'CM.4-0096', 'CM.1-0005.1',
       'CM.1-0019.1', 'CM.1-0021.1', 'CM.1-0030.1', 'CM.4-0085', 'CM.4-0088',
       'CM.1-0002.1', 'CM.1-0013.1', 'CM.1-0020.1', 'CM.1-0022.1',
       'CM.1-0027.1', 'CM.4-0086', 'CM.4-0034', 'CM.4-0009', 'CM.4-0024',
       'CM.4-0037', 'CM.4-0041', 'CM.4-0099', 'CM.1-0006.1', 'CM.1-0008.1'])
gleasonscores = gleasonscores.drop(['CM.4-0014', 'CM.4-0028', 'CM.4-0054', 'CM.4-0055', 'CM.4-0074',
       'CM.4-0078', 'CM.4-0080', 'CM.4-0082', 'CM.4-0093', 'CM.1-0028.1',
       'CM.4-0019', 'CM.4-0064', 'CM.4-0079', 'CM.4-0047', 'CM.4-0048',
       'CM.4-0050', 'CM.4-0051', 'CM.4-0052', 'CM.4-0061', 'CM.4-0076',
       'CM.4-0083', 'CM.4-0087', 'CM.4-0094', 'CM.4-0013', 'CM.4-0067',
       'CM.4-0068', 'CM.4-0070', 'CM.4-0090', 'CM.4-0098', 'CM.4-0045',
       'CM.4-0075', 'CM.4-0081', 'CM.4-0091', 'PT184', 'PT199', 'PT236',
       'PT243', 'CM.4-0066', 'CM.4-0095', 'PT081', 'CMO10311-0036',
       'CM.1-0003.1', 'CM.1-0004.1', 'CM.1-0017.1', 'CM.1-0025.1', 'CM.4-0042',
       'CM.4-0043', 'CM.4-0044', 'CM.4-0046', 'CM.4-0096', 'CM.1-0005.1',
       'CM.1-0019.1', 'CM.1-0021.1', 'CM.1-0030.1', 'CM.4-0085', 'CM.4-0088',
       'CM.1-0002.1', 'CM.1-0013.1', 'CM.1-0020.1', 'CM.1-0022.1',
       'CM.1-0027.1', 'CM.4-0086', 'CM.4-0034', 'CM.4-0009', 'CM.4-0024',
       'CM.4-0037', 'CM.4-0041', 'CM.4-0099', 'CM.1-0006.1', 'CM.1-0008.1'])


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


def predict_null_accuracy(gleason_score):
    print("\n=====================================================\n")
    null_acc = (gleason_score['Gleason'].value_counts().head(1)/len(gleason_score['Gleason']))
    print("Null accurancy score:    ", null_acc)


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
def plotimportances(in_feature_importances, gene_data, num_features, method, threshold, show_meth="save"):
    feature_importances = pd.Series(100*(in_feature_importances/max(in_feature_importances)), index=gene_data.columns)
    if num_features > 50:
        num_features = 50
    fig, ax = plt.subplots(figsize=(15,10))
    top = feature_importances.nlargest(num_features)
    ax.barh(list(reversed(list(top.index))), list(reversed(list(top.data))),  color=['#ff0000' if (x < 50) else '#fff700' if (50 <= x <= 75) else '#33ff00' for x in list(reversed(list(top.data)))])
    ax.set_title("Feature importance: "+method+" (threshold: "+str(threshold)+")")
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("Features (Genes)")
    plt.tight_layout()
    if show_meth == "save":
        plt.savefig(fname = ("figures/importances/"+method+str(threshold)), format='png')
    if show_meth == "show":
        plt.show()
#plotimportances(initial_feature_importance, gene_data, 40, "Initial")

#FIX and impliment
def plot_perm_importances(model, gene_data, gleason_score, method, setname):
    result = permutation_importance(model, gene_data, gleason_score['Gleason'], n_repeats=10,
                            random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=gene_data.columns[sorted_idx])
    ax.set_title("Permutation Importances: "+method+" ("+setname+" set)")
    fig.tight_layout()
    plt.show()




def variance_filter(gene_data, gleason_score, var_thresh=0.0, imp_plot="save"):
    #INFORMATION
    print("\n=====================================================\n")
    method = "Variance filter"
    #remove features with low variance
    print("Removing features with low variance", " (threshold: ", str(var_thresh),")")
    t1 = time.time()
    sel = feature_selection.VarianceThreshold(threshold=var_thresh)
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

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, verbose=True, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    if imp_plot in ("save", "show"):
        print("Plotting importances")
        rfc.fit(gene_data, gleason_score['Gleason'])
        rfc_importances = rfc.feature_importances_
        plotimportances(rfc_importances, gene_data, 50, method, var_thresh, show_meth=imp_plot)
        if rfc.n_features_ < 50:
            plot_perm_importances(rfc, gene_data, gleason_score, method, "train")

    global methodlog
    methodlog.append({"method": method, "threshold": var_thresh, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = variance_filter(gene_data, gleason_score)




def univariate_filter(gene_data, gleason_score, k_val=1000, imp_plot="save"):
    #STATISTIC
    print("\n=====================================================\n")
    method = "Univariate feature selection"
    #univariate feature selection
    #stats tests to select features with highest correlation to target
    print(method, " (n features: ", str(k_val),")")
    #feature extraction
    print("Feature selection")
    t1 = time.time()
    k_best = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=k_val)
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

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, verbose=True, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    if imp_plot in ("save", "show"):
        #checking which are the most important features
        print("Plotting importance")
        rfc.fit(gene_data, gleason_score['Gleason'])
        k_feature_importances = rfc.feature_importances_
        plotimportances(k_feature_importances, gene_data, len(list(gene_data.columns.values)), method, k_val, show_meth=imp_plot)
        if rfc.n_features_ < 50:
            plot_perm_importances(rfc, gene_data, gleason_score, method, "train")

    global methodlog
    methodlog.append({"method": method, "threshold": k_val, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = univariate_filter(gene_data, gleason_score)



def correlation_filter(gene_data, gleason_score, corr_thresh=0.8, imp_plot="save"):
    #PREDICTIVE
    print("\n=====================================================\n")
    method = "High correlation filter"
    #mergedset = pd.merge(gene_data, gleason_score.reset_index(), left_index=True, right_index=True)
    #remove highly correlated features
    #this prevents overfitting (due to highly correlated or colinear features)
    print(method, " (threshold: ", str(corr_thresh),")\n(this may take some time, can crash n>1000)")
    t1 = time.time()
    corr_matrix = gene_data.corr().abs()
    #print(corr_matrix['Gleason'].sort_values(ascending=False).head(10))
    #plot correlation matrix
    matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    #sns.heatmap(matrix)
    #plt.show()

    #find index of feature columns with high correlation
    to_drop = [column for column in matrix.columns if any(matrix[column] > corr_thresh)]
    if "Gleason" in to_drop:
        to_drop.remove("Gleason")
    print("Before:  ", str(gene_data.shape[1]))
    gene_data = gene_data.drop(to_drop, axis=1)
    print("After:   ", str(gene_data.shape[1]))
    print("Removed: ", str(len(to_drop)))

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    #cross validating
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, verbose=True, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    if imp_plot in ("save", "show"):
        #checking which are the most important features
        print("Plotting importance")
        rfc.fit(gene_data, gleason_score['Gleason'])
        correlation_importances = rfc.feature_importances_
        plotimportances(correlation_importances, gene_data, len(list(gene_data.columns.values)), method, corr_thresh, show_meth=imp_plot)
        if rfc.n_features_ < 50:
            plot_perm_importances(rfc, gene_data, gleason_score, method, "train")

    global methodlog
    methodlog.append({"method": method, "threshold": corr_thresh, "size": gene_data.shape[1], "removed": len(to_drop), "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = correlation_filter(gene_data, gleason_score)


def corrlation_with_target(gene_data, gleason_score, corr_thresh=0.05, imp_plot="save"):
    #PREDICTIVE
    print("\n=====================================================\n")
    method = "Target correlation filter"
    print(method, " (threshold: ", str(corr_thresh),")")
    t1 = time.time()

    try:
        gleason_score = gleason_score.reset_index()
    except ValueError:
        pass
    target_correlations = {}

    for feat in list(gene_data.columns.values):
        target_correlations[feat] = abs(gleason_score['Gleason'].corr(gene_data[feat]))

    to_drop = [feat for feat in target_correlations if (target_correlations[feat] < corr_thresh)]
    if "Gleason" in to_drop:
        to_drop.remove("Gleason")
    print("Before:  ", str(gene_data.shape[1]))
    gene_data = gene_data.drop(to_drop, axis=1)
    print("After:   ", str(gene_data.shape[1]))
    print("Removed: ", str(len(to_drop)))

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    #cross validating
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, verbose=True, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    if imp_plot in ("save", "show"):
        #checking which are the most important features
        print("Plotting importance")
        rfc.fit(gene_data, gleason_score['Gleason'])
        correlation_importances = rfc.feature_importances_
        plotimportances(correlation_importances, gene_data, len(list(gene_data.columns.values)), method, corr_thresh, show_meth=imp_plot)
        if rfc.n_features_ < 50:
            plot_perm_importances(rfc, gene_data, gleason_score, method, "train")

    global methodlog
    methodlog.append({"method": method, "threshold": corr_thresh,"size": gene_data.shape[1], "removed": len(to_drop), "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score



def recursive_feature_elimination(gene_data, gleason_score, n_feat=None, step_val=1, imp_plot="save"):
    #GREEDY
    print("\n=====================================================\n")
    method = "Recursive feature elimination"
    #recursive feature elimination
    print(method, " (n features: ", str(n_feat),")")
    t1 = time.time()
    #lr = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=5000) #max_iter specified as does not converge at 1000 (default)
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    #feature extraction
    print("Feature extraction")
    rfe = feature_selection.RFE(rfc, n_features_to_select=n_feat, step=step_val) #, verbose=True)
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

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    if imp_plot in ("save", "show"):
        print("Plotting importance")
        rfc.fit(gene_data, gleason_score['Gleason'])
        recursive_importances = rfc.feature_importances_
        plotimportances(recursive_importances, gene_data, len(list(gene_data.columns.values)), method, n_feat, show_meth=imp_plot)
        if rfc.n_features_ < 50:
            plot_perm_importances(rfc, gene_data, gleason_score, method, "train")

    global methodlogp
    methodlog.append({"method": method, "threshold": n_feat, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = recursive_feature_elimination(gene_data, gleason_score)



def feature_select_from_model(gene_data, gleason_score, thresh=None, imp_plot="save"):
    #MODEL
    print("\n=====================================================\n")
    method = "Model feature select"
    t1 = time.time()
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    #feature selection from model
    #feature extraction
    print("Feature selection from model (threshold: ", str(thresh),")")
    select_model = feature_selection.SelectFromModel(rfc, threshold=thresh)
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

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    if imp_plot in ("save", "show"):
        print("Plotting importance")
        rfc.fit(gene_data, gleason_score['Gleason'])
        featsel_importances = rfc.feature_importances_
        plotimportances(featsel_importances, gene_data, len(list(gene_data.columns.values)), method, thresh, show_meth=imp_plot)
        if rfc.n_features_ < 50:
            plot_perm_importances(rfc, gene_data, gleason_score, method, "train")

    global methodlog
    methodlog.append({"method": method, "threshold": thresh, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = feature_select_from_model(gene_data, gleason_score)



def tree_based_selection(gene_data, gleason_score, thresh=None, imp_plot="save"):
    #MODEL
    print("\n=====================================================\n")
    method = "Tree based selection"
    #tree based selection
    print(method, " (threshold: ", str(thresh),")")
    t1 = time.time()
    etc = ExtraTreesClassifier(n_estimators=100)
    print("Fitting")
    etc = etc.fit(gene_data, gleason_score['Gleason'])
    etc_model = feature_selection.SelectFromModel(etc, threshold=thresh, prefit=True)
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

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    if imp_plot in ("save", "show"):
        print("Plotting importances")
        etc_importances = ExtraTreesClassifier(n_estimators=100).fit(gene_data, gleason_score['Gleason']).feature_importances_
        plotimportances(etc_importances, gene_data, len(list(gene_data.columns.values)), method, thresh, show_meth=imp_plot)
        rfc.fit(gene_data, gleason_score['Gleason'])
        if rfc.n_features_ < 50:
            plot_perm_importances(rfc, gene_data, gleason_score, method, "train")

    global methodlog
    methodlog.append({"method": method, "threshold": thresh, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = tree_based_selection(gene_data, gleason_score, mergedset)


def L1_based_select(gene_data, gleason_score, thresh=None, imp_plot="save"):
    #MODEL
    print("\n=====================================================\n")
    method = "L1-based selection"
    print(method, " (threshold: ", str(thresh),")")
    #lr = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=5000)
    print("Fitting")
    t1 = time.time()
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(gene_data, gleason_score['Gleason'])
    L1_model = feature_selection.SelectFromModel(lsvc, threshold=thresh, prefit=True)
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

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    if imp_plot in ("save", "show"):
        print("Plotting importances")
        rfc.fit(gene_data, gleason_score['Gleason'])
        L1_importance = rfc.feature_importances_
        plotimportances(L1_importance, gene_data, len(list(gene_data.columns.values)), method, thresh, show_meth=imp_plot)
        if rfc.n_features_ < 50:
            plot_perm_importances(rfc, gene_data, gleason_score, method, "train")

    global methodlog
    methodlog.append({"method": method, "threshold": thresh, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score

#gene_data, gleason_score = L1_based_select(gene_data, gleason_score)











#still not sure, needs fixing, works in pipeline
def PCA_analysis(gene_data, gleason_score, n_comp):
    #REDUNDANCY
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

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)

    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(something.columns.values)], gleasonscores['Gleason'], cv=5)

    print("RFC Scores:  ", rfc_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    global methodlog
    methodlog.append({"method": method, "threshold": n_comp, "size": gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    #return gene_data, gleason_score

#PCA_analysis(gene_data, gleason_score)






def basic_method_iteration(gene_train, gleason_train):
    global methodlog
    func_list = [variance_filter,#(gene_train, gleason_train, var_thresh=0.0)
                 univariate_filter,#(gene_train_filt, gleason_train, k_val=1000)
                 correlation_filter,#(gene_train_filt, gleason_train, corr_thresh=0.75)
                 corrlation_with_target,#(gene_train_filt, gleason_train, corr_thresh=0.1)
                 recursive_feature_elimination,#(gene_train_filt, gleason_train, n_feat=None)
                 feature_select_from_model,#(gene_train_filt, gleason_train, thresh=None)
                 tree_based_selection,#(gene_train_filt, gleason_train, thresh=None)
                 L1_based_select]#(gene_train_filt, gleason_train, thresh=None)
    meth_func_index = {"Variance filter":variance_filter, "Univariate feature selection":univariate_filter, "High correlation filter":correlation_filter,
                        "Target correlation filter":corrlation_with_target, "Recursive feature elimination":recursive_feature_elimination,
                        "Model feature select":feature_select_from_model, "Tree based selection":tree_based_selection, "L1-based selection":L1_based_select}
    func_meth_index = {variance_filter:"Variance filter", univariate_filter:"Univariate feature selection", correlation_filter:"High correlation filter",
                        corrlation_with_target:"Target correlation filter", recursive_feature_elimination:"Recursive feature elimination",
                        feature_select_from_model:"Model feature select", tree_based_selection:"Tree based selection", L1_based_select:"L1-based selection"}
    thresholds_dictionary ={variance_filter: np.linspace(0, 0.05, 5), #var_thresh=0.0
                 univariate_filter: np.linspace(500, 2500, 5, dtype=int), #k_val=1000)
                 correlation_filter: np.linspace(0.5, 0.9, 9), #corr_thresh=0.75)
                 corrlation_with_target: np.linspace(0.05, 0.25, 5), #corr_thresh=0.1)
                 recursive_feature_elimination: [None], #np.linspace(),#n_feat=None)
                 feature_select_from_model: [None], #np.linspace(), #thresh=None)
                 tree_based_selection: [None], #np.linspace(): #thresh=None)
                 L1_based_select: [None]} #np.linspace()} #thresh=None)
    topnfun, topscore, topvar = 0, 0, 100
    gene_train_filt = gene_train.copy()
    optcheck, opttaken, done_list = [], [], []
    for i in range(len(func_list)):
        print("ROUND: ", i)
        func_to_do = list(set(func_list)-set(done_list))
        for x, fun in enumerate(func_to_do):
            print("\n=====================================================\n")
            print("ROUND: ", i)
            print("METHOD: ", x, func_meth_index[fun])
            if len(list(gene_train_filt.columns.values)) > 1000 and fun in (correlation_filter, recursive_feature_elimination):
                methodlog.append({"method": func_meth_index[fun], "threshold": 0, "size": len(list(gene_train_filt.columns.values)), "removed": 0, "rfc_scores": [0,0,0,0,0]})
                print("SKIPPING")
                next(iter(func_to_do))
            else:
                for v in thresholds_dictionary[fun]:
                    if fun == univariate_filter:
                        if v < len(list(gene_train_filt.columns.values)):
                            gene_train_filt_temp, gleason_train = fun(gene_train_filt, gleason_train, v, imp_plot=False)
                    else:
                        gene_train_filt_temp, gleason_train = fun(gene_train_filt, gleason_train, v, imp_plot=False)
            if x == len(func_to_do)-1:
                print("\n\n\nROUND", i, "FINISHED\n\n\n")
                optcheck.append(methodlog)
                print(len(optcheck), len(optcheck[i]))
                methodlog = []
                for method in optcheck[i]:
                    print("\n=====================================================\n")
                    print("Checking method: ", optcheck[i].index(method), "/", len(optcheck[i])-1, i, method['method'], method['threshold'])
                    tempscore = np.mean(method['rfc_scores'])
                    tempvar = np.std(method['rfc_scores'])*2
                    print("Score: ", tempscore, " Var: ", tempvar)
                    print("Removed: ", method['removed'])
                    if tempscore > topscore and tempvar <= topvar+(tempscore-topscore):
                        topscore = tempscore.copy()
                        topvar = tempvar.copy()
                        topmethodlog = method.copy()
                        topnfun = i
                        print("Current best!!")
                    if optcheck[i].index(method) == len(optcheck[i])-1:
                        print("First:", topnfun, i)
                        if topnfun == i:
                            print("Second:", topnfun, i)
                            gene_train_filt, gleason_train = meth_func_index[topmethodlog['method']](gene_train_filt, gleason_train, topmethodlog['threshold'])
                            opttaken.append(topmethodlog)
                            print("\n=====================================================\n")
                            print("\n\n\n\nTOP METHOD:  ", topmethodlog['method'], "\n")
                            print("TOP SCORE:   ", topscore, " BEST VAR:    ", topvar, "\n")
                            if i != 0:
                                print("Improvement: ", topscore-np.mean(opttaken[i-1]['rfc_scores']), "              ", topvar-(np.std(opttaken[i-1]['rfc_scores'])*2), "\n")
                            print("Removed:     ", topmethodlog['removed'], "New size:   ", topmethodlog['size'], "\n")
                            print("Route:       ", [(order['method'], order['threshold']) for order in opttaken], "\n\n\n\n")
                            print("\n=====================================================\n")
                            if topmethodlog['method'] != "Recursive feature elimination":
                                done_list.append(meth_func_index[topmethodlog['method']])
                        else:
                            print("\n=====================================================\n")
                            print("\n\n\n\nDIDN'T BEAT LAST ROUND")
                            print("Total number of rounds: ", topnfun, "\n")
                            print("TOP SCORE:   ", topscore, " BEST VAR:    ", topvar, "\n")
                            print("Removed:     ", len(genedata.columns.values)-len(gene_train_filt.columns.values), "New size:   ", topmethodlog['size'], "\n")
                            print("Route:       ", [(order['method'], order['threshold']) for order in opttaken], "\n\n\n\n")
                            print("\n=====================================================\n")
                            return gene_train_filt, opttaken











def split_method_iteration(gene_train, gleason_train):
    global methodlog
    info_func_list = [variance_filter,#(gene_train, gleason_train, var_thresh=0.0)
                      univariate_filter]#(gene_train_filt, gleason_train, k_val=1000)
    corr_func_list = [correlation_filter,#(gene_train_filt, gleason_train, corr_thresh=0.75)
                      corrlation_with_target]#(gene_train_filt, gleason_train, corr_thresh=0.1)
    gred_func_list = [recursive_feature_elimination]#(gene_train_filt, gleason_train, n_feat=None)
    modl_func_list = [feature_select_from_model,#(gene_train_filt, gleason_train, thresh=None)
                      tree_based_selection,#(gene_train_filt, gleason_train, thresh=None)
                      L1_based_select]#(gene_train_filt, gleason_train, thresh=None)
    list_list = [info_func_list, corr_func_list, gred_func_list, modl_func_list]
    meth_func_index = {"Variance filter":variance_filter, "Univariate feature selection":univariate_filter, "High correlation filter":correlation_filter,
                        "Target correlation filter":corrlation_with_target, "Recursive feature elimination":recursive_feature_elimination,
                        "Model feature select":feature_select_from_model, "Tree based selection":tree_based_selection, "L1-based selection":L1_based_select}
    func_meth_index = {variance_filter:"Variance filter", univariate_filter:"Univariate feature selection", correlation_filter:"High correlation filter",
                        corrlation_with_target:"Target correlation filter", recursive_feature_elimination:"Recursive feature elimination",
                        feature_select_from_model:"Model feature select", tree_based_selection:"Tree based selection", L1_based_select:"L1-based selection"}
    thresholds_dictionary ={variance_filter: np.linspace(0, 0.05, 5), #var_thresh=0.0
                 univariate_filter: np.linspace(500, 2500, 5, dtype=int), #k_val=1000)
                 correlation_filter: np.linspace(0.5, 0.9, 9), #corr_thresh=0.75)
                 corrlation_with_target: np.linspace(0.05, 0.25, 5), #corr_thresh=0.1)
                 recursive_feature_elimination: [None], #np.linspace(),#n_feat=None)
                 feature_select_from_model: [None], #np.linspace(), #thresh=None)
                 tree_based_selection: [None], #np.linspace(): #thresh=None)
                 L1_based_select: [None]} #np.linspace()} #thresh=None)
    topnfun, topscore, topvar, skipped = 0, 0, 100, 0
    gene_train_filt = gene_train.copy()
    optcheck, opttaken, done_list = [], [], []
    for i in range(len(list_list)):
        print("ROUND: ", i)
        func_to_do = list(set(list_list[i])-set(done_list))
        for x, fun in enumerate(func_to_do):
            print("\n=====================================================\n")
            print("ROUND: ", i)
            print("METHOD: ", x, func_meth_index[fun])
            if len(list(gene_train_filt.columns.values)) > 1000 and fun in (correlation_filter, recursive_feature_elimination):
                methodlog.append({"method": func_meth_index[fun], "threshold": 0, "size": len(list(gene_train_filt.columns.values)), "removed": 0, "rfc_scores": [0,0,0,0,0]})
                print("SKIPPING")
                next(iter(func_to_do))
            else:
                for v in thresholds_dictionary[fun]:
                    if fun == univariate_filter:
                        if v < len(list(gene_train_filt.columns.values)):
                            gene_train_filt_temp, gleason_train = fun(gene_train_filt, gleason_train, v, imp_plot=False)
                    else:
                        gene_train_filt_temp, gleason_train = fun(gene_train_filt, gleason_train, v, imp_plot=False)
            if x == len(func_to_do)-1:
                print("\n\n\nROUND", i, "FINISHED\n\n\n")
                optcheck.append(methodlog)
                print(len(optcheck))
                methodlog = []
                for method in optcheck[i]:
                    print("\n=====================================================\n")
                    print("Checking method: ", optcheck[i].index(method), i, method['method'], method['threshold'])
                    tempscore = np.mean(method['rfc_scores'])
                    tempvar = np.std(method['rfc_scores'])*2
                    print("Score: ", tempscore, " Var: ", tempvar)
                    print("Removed: ", method['removed'])
                    if tempscore > topscore and tempvar <= topvar+(tempscore-topscore):
                        topscore = tempscore.copy()
                        topvar = tempvar.copy()
                        topmethodlog = method.copy()
                        topnfun = i
                        print("Current best!!")
                    if optcheck[i].index(method) == len(optcheck[i])-1:
                        if topnfun == i:
                            gene_train_filt, gleason_train = meth_func_index[topmethodlog['method']](gene_train_filt, gleason_train, topmethodlog['threshold'])
                            opttaken.append(topmethodlog)
                            print("\n=====================================================\n")
                            print("\n\n\n\nTOP METHOD:  ", topmethodlog['method'], "\n")
                            print("TOP SCORE:   ", topscore, " BEST VAR:    ", topvar, "\n")
                            if i != 0:
                                print("Improvement: ", topscore-np.mean(opttaken[i-1]['rfc_scores']), "              ", topvar-(np.std(opttaken[i-1]['rfc_scores'])*2), "\n")
                            print("Removed:     ", topmethodlog['removed'], "New size:   ", topmethodlog['size'], "\n")
                            print("Route:       ", [(order['method'], order['threshold']) for order in opttaken], "\n\n\n\n")
                            print("\n=====================================================\n")
                            if topmethodlog['method'] != "Recursive feature elimination":
                                done_list.append(meth_func_index[topmethodlog['method']])
                        else:
                            if i != len(list_list)-1:
                                next(iter(list_list))
                                skipped += 1
                            if i == len(list_list)-1:
                                print("\n=====================================================\n")
                                print("\n\n\n\nDIDN'T BEAT LAST ROUND")
                                print("Total number of rounds: ", topnfun-skipped, "\n")
                                print("TOP SCORE:   ", topscore, " BEST VAR:    ", topvar, "\n")
                                print("Removed:     ", len(genedata.columns.values)-len(gene_train_filt.columns.values), "New size:   ", topmethodlog['size'], "\n")
                                print("Route:       ", [(order['method'], order['threshold']) for order in opttaken], "\n\n\n\n")
                                print("\n=====================================================\n")
                                return gene_train_filt, opttaken



def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))





#print(methodlog)

def visualise_accuracy_methodlog(methodlog, show_meth="show"):
    methodorder = []
    #size = []
    rfc_accuracy = []
    rfc_error = []
    for i in range(len(methodlog)):
        methodorder.append(methodlog[i]['method'])
        #size.append(methodlog[i]['size'])
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
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross validation scores\n for RFC')
    ax.set_xticks(ind) #+ width / 2)
    ax.set_xticklabels(tuple(methodorder))
    #ax.legend( (rects1[0]), ('Random Forest Classifier') ) #, rects2[0]), ('Logistic Regression', 'Random Forest Classifier') )
    if show_meth == "save":
        plt.savefig("Score vis")
    if show_meth == "show":
        plt.show()



def vis_trees(model, name):
    #assuming model is RFC
    #export_graphviz(model.base_estimator_, out_file="trees/tree_base_"+str(name)+".dot")
    #call(['dot', '-Tpng', 'trees/tree_base_'+str(name)+'.dot', '-o', 'trees/tree_base_'+str(name)+'.png', '-Gdpi=600'])
    for i in range(len(model.estimators_)):
        export_graphviz(model.estimators_[i], out_file="figures/trees/tree_"+str(name)+"_"+str(i)+".dot")
        call(['dot', '-Tpng', 'figures/trees/tree_'+str(name)+'_'+str(i)+'.dot', '-o', 'figures/trees/tree_'+str(name)+'_'+str(i)+'.png', '-Gdpi=600'])



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



def plot_learning_curve(estimator, gene_data, gleason_score, ylim=None, cv=None,
                    n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title("Learning curve for Random Forest Classifier")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, gene_data, gleason_score['Gleason'], cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()











"""
#Relative Feature Importance, been updated to just feature importance
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
"""



"""
def multilabel_confusion_plot(model, y_test, pred): #redundent after update
    target_names = list(model.classes_)
    print(target_names)
    cm = multilabel_confusion_matrix(y_test, pred)
    print(cm)
    cm = np.reshape(cm, (len(target_names), len(target_names)))
    print(cm)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Multilabel Confusion Matrix')
    plt.colorbar()

    x_tick_marks = np.arange(len(target_names))
    y_tick_marks = np.arange(len(target_names))
    plt.xticks(x_tick_marks, target_names, rotation=45)
    plt.yticks(y_tick_marks, target_names)

    normalize = True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted labelaccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
"""




"""
    labels = model.classes_
    cm = multilabel_confusion_matrix(y_test, pred)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
"""



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
methodlog.append({"method": method, "threshold": thresh, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "lr_scores": lr_scores, "rfc_scores": rfc_scores, "lr_acc": lr_scores.mean(), "rfc_acc": rfc_scores.mean(), "lr_err": lr_scores.std()*2, "rfc_err": rfc_scores.std()*2})
"""

"""
from mdr import MDR #only supports binary endpoint
def MDR_select(gene_data, gleason_score, thresh=None, imp_plot="save"):
    #MODEL
    print("\n=====================================================\n")
    method = "MDR selection"
    print(method, " (threshold: ", str(thresh),")")
    #lr = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=5000)
    print("Fitting")
    t1 = time.time()
    mdrm = MDR()
    mdr_fit = mdrm.fit(gene_data, gleason_score['Gleason'])
    #L1_importances = lsvc.feature_importances_ #not supported with current method
    #plotimportances(L1_importances, gene_data, len(L1_importances), method)
    print("Transforming")
    model_features = mdrm.transform(gene_data)

    gene_names = list(gene_data.columns.values)
    before_num_genes = gene_data.shape[1]
    passed_MDR_filter = MDR_model.get_support()
    print("Before:  ", str(before_num_genes))
    MDR_filter_genes = []
    for bool, gene in zip(passed_MDR_filter, gene_names):
        if bool:
            MDR_filter_genes.append(gene)

    gene_data = pd.DataFrame(model_features, columns=L1_filter_genes)
    #gene_data = gene_data.set_index(row_names)
    print("After:   ", str(gene_data.shape[1]), str(np.bool(gene_data.shape[1]==model_features.shape[1])))
    print("Removed: ", str(before_num_genes-gene_data.shape[1]))

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Cross validating")
    #print(cross_validate(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5))
    rfc_scores = cross_val_score(rfc, genedata[list(gene_data.columns.values)], gleasonscores['Gleason'], cv=5, scoring='accuracy')
    print("RFC Scores:  ", rfc_scores)
    print("Accuracy:    %0.2f (+/- %0.2f)" % (rfc_scores.mean(), rfc_scores.std() * 2))

    if imp_plot in ("save", "show"):
        print("Plotting importances")
        rfc.fit(gene_data, gleason_score['Gleason'])
        MDR_importance = rfc.feature_importances_
        plotimportances(MDR_importance, gene_data, len(list(gene_data.columns.values)), method, thresh, show_meth=imp_plot)
        if rfc.n_features_ < 50:
            plot_perm_importances(rfc, gene_data, gleason_score, method, "train")

    global methodlog
    methodlog.append({"method": method, "threshold": thresh, "size": gene_data.shape[1], "removed": before_num_genes-gene_data.shape[1], "rfc_scores": rfc_scores, "time-taken": time.time()-t1})

    print("\n=====================================================\n")
    return gene_data, gleason_score
"""
