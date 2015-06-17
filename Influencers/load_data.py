import numpy as np
import pandas as pd

train_path = "data/train.csv"
test_path = "data/test.csv"

def load( type ):

	if type == "train":
		data = pd.read_csv(train_path)
		l_limit = 1
		r_limit = 12
	elif type == "test":
		data = pd.read_csv(test_path)
		l_limit = 0
		r_limit = 11
	else:
		print "error"
		return 0;	

	if type == "train":
		y = data[['Choice']]

	X_A = data.ix[:,l_limit:r_limit]
	X_B = data.ix[:,r_limit:]

	#Return panda files
	if type == "train":
		return X_A, X_B, y
	else:
		return X_A, X_B