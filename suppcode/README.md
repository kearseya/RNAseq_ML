## Learning curve

Exmperimental learning curve (*NB needs correcting*):

![learning curve](/figs/learning_curve_example_bitbetter.png)


## Comparing reduction

Authors: Robert McGibbon, Joel Nothman

This example constructs a pipeline that does dimensionality
reduction followed by prediction with a support vector
classifier. It demonstrates the use of GridSearchCV and
Pipeline to optimize over different classes of estimators in a
single CV run -- unsupervised PCA and NMF dimensionality
reductions are compared to univariate feature selection during
the grid search.

Comparing of reduction has been achieved using a modified version of Robert McGibbon and Joel Nothman's code (source: https://scikit-learn.org/0.18/auto_examples/plot_compare_reduction.html)

![comparing reduction](/figs/compare_reduction.png)

*NB there are errors when code is ran, failed to converge on some steps*
