import numpy as np

train_path = "data/train.csv"
test_path = "data/test.csv"

def load( type ):

	if type == "train":
		filee = open(train_path)
		l_limit = 1
		r_limit = 12
	elif type == "test":
		filee = open(test_path)
		l_limit = 0
		r_limit = 11
	else:
		print "error"
		return 0;

	# Save in header the features label
	header = filee.next().rstrip().split(',')	

	y = []
	X_A = []
	X_B = []

	for line in filee:
		splitted = line.rstrip().split(',')
		
		if type == "train":
			label = int(splitted[0])

		A_features = [float(item) for item in splitted[l_limit:r_limit]]
		B_features = [float(item) for item in splitted[r_limit:]]

		if type == "train":
			y.append(label)

		X_A.append(A_features)
		X_B.append(B_features)
	filee.close()

	#Convert train sets to numpy arrays
	if type == "train":
		y = np.array(y)

	X_A = np.array(X_A)
	X_B = np.array(X_B)

	if type == "train":
		return X_A, X_B, y
	else:
		return X_A, X_B