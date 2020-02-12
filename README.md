# RNAseq_ML

Draft script for discovering best methods for feature selection with RNAseq data (using sklearn package).

Mehods include:
* Removing low variance
* Univariate filter
* Correlation filter
* Recursive feature elimination
* Feature selection from model
* PCA filter


Data from:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54460

Best result (var, uni, cor, rec, sel, pca): 
* LR  Scores:    [0.68181818 0.76190476 0.76190476 0.80952381 0.76190476] 
* Accuracy: 0.76 (+/- 0.08) 
* RFC Scores:   [0.68181818 0.80952381 0.76190476 0.85714286 0.80952381] 
* Accuracy: 0.78 (+/- 0.12) 
