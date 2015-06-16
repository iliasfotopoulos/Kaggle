from sklearn import linear_model
from sklearn import metrics
import numpy as np
import load_data

#X_train_A, X_train_B, y_train = load_data.load("train")

def transform_features(X):
	return np.log(1+X)

'''
def followers_diff(X):
	X[:,0] = X[:,0] - X[:,1]
	return np.delete(X,1,1)
'''

def boost(X):
	X[:,0] = X[:,0] * 2
	return X

def proc(X):
	X = transform_features( boost(X) )
	return X


