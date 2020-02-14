# RNAseq_ML

Investigating methods to undertake feature selection and reduction on RNA-seq data.  

Data from:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54460


## Next step is to use the sklearn pipeline functions and hyperparameter finding

Manual methods script for initial investigation of a few methods, sklearn submodule pipeline has wider functionality. Hyperparameters also need to be found to make model more optimal.

### Preliminary data generated from manual methods script

Draft scripts for discovering best methods for feature selection with RNAseq data (using sklearn package). 
To run:  
`$ python3 test_script.py`

Mehods include:
* Removing low variance
* Univariate filter
* Correlation filter
* Recursive feature elimination
* Feature selection from model
* Tree based selection
* L1-based selection

Also PCA analysis


Example cross-validation (var -> uni, 500 features): 
* RFC Scores:   [0.72727273 0.80952381 0.78947368 0.78947368 0.78947368]
* Accuracy:     0.78 (+/- 0.06)


Validation curve:
![validation curve example](/figs/validation_curve_example.png)

Feature LR and RFC scores are visualised at the end:
![cross validation scores example](/figs/cross_val_graph.png)

For the correlation filter, a heatmap is generated:
![heatplot example](/figs/correlation_matrix_example.png)

Feature importance is also extracted and plotted at some steps:
![relative feature importance example](/figs/feature_importance_example.png)
