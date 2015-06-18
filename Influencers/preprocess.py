from sklearn import linear_model
from sklearn import metrics
import pandas as pd
import numpy as np

#X_train_A, X_train_B, y_train = load_data.load("train")

def common_name(A,B):
	A.rename(columns={ 'A_follower_count':'follower_count',
 'A_following_count':'following_count',
 'A_listed_count':'listed_count',
 'A_mentions_received':'mentions_received',
 'A_retweets_received':'retweets_received',
 'A_mentions_sent':'mentions_sent',
 'A_retweets_sent':'retweets_sent',
 'A_posts':'posts',
 'A_network_feature_1':'network_feature_1',
 'A_network_feature_2':'network_feature_2',
 'A_network_feature_3':'network_feature_3'}, inplace= True)
	B.rename(columns={ 'B_follower_count':'follower_count',
 'B_following_count':'following_count',
 'B_listed_count':'listed_count',
 'B_mentions_received':'mentions_received',
 'B_retweets_received':'retweets_received',
 'B_mentions_sent':'mentions_sent',
 'B_retweets_sent':'retweets_sent',
 'B_posts':'posts',
 'B_network_feature_1':'network_feature_1',
 'B_network_feature_2':'network_feature_2',
 'B_network_feature_3':'network_feature_3'}, inplace= True)
	return A,B


def log(X):
	return X.apply(np.log1p)

def retweet_ratio(X):
	X.ix[:,'posts'].replace(0,1,inplace=True)
	X["retweet_ratio"] = ( X.ix[:,'retweets_received'] / X.ix[:,'posts'] ) * 1.5
	return X

def follow_ratio(X):
	X.ix[:,1].replace(0,1,inplace=True)
	X["follow_ratio"] = (X.ix[:,0] / X.ix[:,1] ) *0.1
	return X

def follow_dif(X):
	X["follow_dif"] = (X.ix[:,'follower_count'] - X.ix[:,'following_count'])	
	X["follow_dif"].replace(0,1,inplace=True)
	#X.drop('follower_count', axis=1, inplace=True)
	#X.drop('following_count', axis=1, inplace=True)
	return X

def proc(X):
	#Add new features
	X = follow_ratio(X)
	X = retweet_ratio(X)
	return log(X)