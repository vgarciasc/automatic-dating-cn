import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

def get_baseline_df():
	df = pd.read_csv('data/baseline/baseline_dataset_ALL_STOPWORDS.csv', sep=';', index_col=0)

	s19 = df[df.century == '19th Century'].index
	s18 = df[df.century == '18th Century'].index
	sec = s19.append(s18)
	df = df.iloc[sec,:]

	X = df.iloc[:,5:]
	y = df.iloc[:,4]
	y = 1 * (y == '19th Century') # deixar y binário -> séc XIX=1
	return X, y, df

def get_network_df():
	df = pd.read_csv('data/network/network_metrics.csv', sep=';', index_col=0)

	s19 = df[df.century == '19th Century'].index
	s18 = df[df.century == '18th Century'].index
	sec = s19.append(s18)
	df = df.iloc[sec,:]

	X = df.iloc[:,13:]
	y = df.iloc[:,5]
	y = 1 * (y == '19th Century') # deixar y binário -> séc XIX=1
	return X, y, df

def logreg(X_train, y_train):
	logreg_clf = LogisticRegression().fit(X_train, y_train)
	y_train_pred = logreg_clf.predict(X_train)
	y_test_pred = logreg_clf.predict(X_test)
	return y_train_pred, y_test_pred

def knn(X_train, y_train):
	knn = KNeighborsClassifier()
	param_grid = {'n_neighbors': np.arange(1, 25)}
	knn_gscv = GridSearchCV(knn, param_grid, cv=5)
	knn_gscv.fit(X_train, y_train)
	y_test_pred = knn_gscv.predict(X_test)
	y_train_pred = knn_gscv.predict(X_train)
	return y_train_pred, y_test_pred

def print_metrics(y_train, y_test, y_train_pred, y_test_pred):
	print('\tPrecisão de treinamento: '+ str(metrics.precision_score(y_train, y_train_pred, average=avg)))
	print('\tPrecisão de teste: '+ str(metrics.precision_score(y_test, y_test_pred, average=avg)))
	print('\tSensibilidade de treinamento: '+ str(metrics.recall_score(y_train, y_train_pred, average=avg)))
	print('\tSensibilidade de teste: '+ str(metrics.recall_score(y_test, y_test_pred, average=avg)))

if __name__ == '__main__':
	X, y, df = get_baseline_df()
	# X, y, df = get_network_df()

	avg = 'weighted'

	# STRATIFIED

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify = df['century'])

	print("Regressão logística:")
	y_train_pred, y_test_pred = logreg(X_train, y_train)
	print_metrics(y_train, y_test, y_train_pred, y_test_pred)

	print("KNN:")
	y_train_pred, y_test_pred = knn(X_train, y_train)
	print_metrics(y_train, y_test, y_train_pred, y_test_pred)

	# LEAVE-ONE-OUT

	logreg_training_precisions = []
	logreg_test_precisions = []
	logreg_training_recalls = []
	logreg_test_recalls = []

	knn_training_precisions = []
	knn_test_precisions = []
	knn_training_recalls = []
	knn_test_recalls = []

	for train_index, test_index in LeaveOneOut().split(X):
		X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
		y_train, y_test = y.values[train_index], y.values[test_index]
		
		y_train_pred, y_test_pred = logreg(X_train, y_train)

		logreg_training_precisions.append(metrics.precision_score(y_train, y_train_pred, average=avg))
		logreg_test_precisions.append(metrics.precision_score(y_test, y_test_pred, average=avg))
		logreg_training_recalls.append(metrics.recall_score(y_train, y_train_pred, average=avg))
		logreg_test_recalls.append(metrics.recall_score(y_test, y_test_pred, average=avg))
		
		y_train_pred, y_test_pred = knn(X_train, y_train)
		
		knn_training_precisions.append(metrics.precision_score(y_train, y_train_pred, average=avg))
		knn_test_precisions.append(metrics.precision_score(y_test, y_test_pred, average=avg))
		knn_training_recalls.append(metrics.recall_score(y_train, y_train_pred, average=avg))
		knn_test_recalls.append(metrics.recall_score(y_test, y_test_pred, average=avg))

	print("Regressão logística:")
	print("\tPrecisão média de treinamento:", np.mean(logreg_training_precisions))
	print("\tPrecisão média de teste:", np.mean(logreg_test_precisions))
	print("\tSensibilidade média de treinamento:", np.mean(logreg_training_recalls))
	print("\tSensibilidade média de teste:", np.mean(logreg_test_recalls))

	print("KNN:")
	print("\tPrecisão média de treinamento:", np.mean(knn_training_precisions))
	print("\tPrecisão média de teste:", np.mean(knn_test_precisions))
	print("\tSensibilidade média de treinamento:", np.mean(knn_training_recalls))
	print("\tSensibilidade média de teste:", np.mean(knn_test_recalls))