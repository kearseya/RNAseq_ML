# RNAseq_ML

Investigating methods to undertake feature selection and reduction on RNA-seq data.  

Data from: 
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54460
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE54460&format=file&file=GSE54460%5FFPKM%2Dgenes%2DTopHat2%2D106samples%2D12%2D4%2D13%2Etxt%2Egz


A large number of Gleason score 7s (80/105 -> 10/35) have been temp removed as they were causing bias.

### Preliminary data generated from manual methods script

Draft scripts for discovering best methods for feature selection with RNAseq data (using sklearn package). 
To run:  
`$ python3 test_script.py`

Mehods include:
* Removing low variance
* Univariate filter
* High correlated features filter
* Low target correlation filter
* Recursive feature elimination
* Feature selection from model
* Tree based selection
* L1-based selection

Also PCA analysis


Multilabel confusion matrix (normalised for true data):  
![MlCM](/example_figs/multilabel_confusion_matrix_unbias.png)

Validation curve:  
![validation curve example](/example_figs/validation_curve_example_unbias.png)

Feature cross validation scores are visualised in order of method used for feature selection at the end:
![cross validation scores example](/example_figs/cross_val_graph_progress_through_methods.png)

For the high correlation filter, a heatmap is generated:
![heatplot example](/example_figs/correlation_matrix_example.png)

Feature importance is also extracted and plotted at each step:
![relative feature importance example](/example_figs/feature_importance_example.png)
