from sklearn import linear_model
from sklearn import metrics
import numpy as np
import load_data

X_train_A, X_train_B, y_train = load_data.load("train")
X_test_A, X_test_B = load_data.load("test")

#Preprocess
def transform_features(x):
	return np.log(1+x)

X_train = transform_features(X_train_A) - transform_features(X_train_B)

model = linear_model.LogisticRegression(fit_intercept=False)
model.fit(X_train,y_train)

preds = model.predict_proba(X_train)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_train, preds)
auc = metrics.auc(fpr,tpr)

#print 'AuC score on training data:',metrics.roc_auc_score(y_train,preds.T)
print 'AuC score on training data:',auc

###########################
# PREDICTING ON TEST DATA
###########################
X_test = transform_features(X_test_A) - transform_features(X_test_B)
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