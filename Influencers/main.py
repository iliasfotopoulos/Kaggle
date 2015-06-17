from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import load_data as load_data
import preprocess as pre

X_train_A, X_train_B, y_train = load_data.load("train")
X_test_A, X_test_B = load_data.load("test")

X_train_A, X_train_B = pre.common_name(X_train_A,X_train_B)
X_train = pre.proc(X_train_A) - pre.proc(X_train_B)

model = linear_model.LogisticRegression(fit_intercept=False)
#params = {'n_estimators':200, 'learning_rate':0.1,'max_depth':3, 'random_state':0}
#model = GradientBoostingClassifier(**params)
model.fit(X_train,y_train['Choice'])

preds = model.predict_proba(X_train)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_train, preds)
auc = metrics.auc(fpr,tpr)

#print 'AuC score on training data:',metrics.roc_auc_score(y_train,preds.T)
print 'AuC score on training data:',auc

###########################
# PREDICTING ON TEST DATA
###########################
X_test_A, X_test_B = pre.common_name(X_test_A,X_test_B)
X_test = pre.proc(X_test_A) - pre.proc(X_test_B)
preds_test = model.predict_proba(X_test)[:,1]

###########################
# WRITING SUBMISSION FILE
###########################
header = np.array(["Id","Choice"])
predfile = open('predictions.csv','w+')

i = 1;
print >>predfile,','.join(header)
for line in preds_test:
	print >>predfile, str(i)+','+str(line)
	i = i+1

predfile.close()