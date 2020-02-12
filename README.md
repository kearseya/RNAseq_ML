# RNAseq_ML

Investigating methods to undertake feature selection and reduction on RNA-seq data.  

Data from:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54460


## Next step is to use the sklearn pipeline functions

Manual methods script for initial investigation of a few methods, sklearn submodule pipeline has wider functionality.

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
* PCA filter


Best result (var, uni, cor, rec, sel, pca): 
* LR  Scores:     [0.68181818 0.76190476 0.76190476 0.80952381 0.76190476] 
* Accuracy:       0.76 (+/- 0.08) 
* RFC Scores:     [0.68181818 0.80952381 0.76190476 0.85714286 0.80952381] 
* Accuracy:       0.78 (+/- 0.12) 

Feature LR and RFC scores are visualised at the end:
![cross validation scores](/figs/cross_val_graph.png)

For the correlation filter, a heatmap is generated:
![heatplot](/figs/correlation_matrix_example.png)

Feature importance is also extracted and plotted at some steps:
![relative feature importance](/figs/feature_importance_example.png)
