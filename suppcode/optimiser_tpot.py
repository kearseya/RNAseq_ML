from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import sys

from test_script import genedata, gleasonscores, gene_train_filt

X_train, X_test, y_train, y_test = train_test_split(genedata[list(gene_train_filt.columns.values)], gleasonscores,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train['Gleason'])
print(tpot.score(X_test, y_test['Gleason']))
tpot.export(str(sys.argv[1])+'_pipeline.py')
