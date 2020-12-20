from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle

import numpy as np
import pandas as pd
import sys
import json


def get_confMatrix(y_act, y_pred):
	"""
	Constructs a confusion matrix containing total TP FP TN and FN for each class
	Parameters:
	===========
	y_act: 2D array, binarized actual labels
	y_pred: 2D array, binarized predicted labels

	Returns:
	conf_matrix: 2D array, (n_labels, 4) containing TP FP TN and FN for each class
	"""

	conf = [] #TP, FP, TN, FN
	assert y_act.shape == y_pred.shape
	for i in range(y_act.shape[1]):
		y_act_samp = y_act[:,i]
		y_pred_samp = y_pred[:,i]
		conf_array = [0]*4
		for j in range(len(y_act_samp)):
			if y_act_samp[j] == 0:
				if y_pred_samp[j] == 0:
					conf_array[2] += 1
				else:
					conf_array[1] += 1
			else:
				if y_pred_samp[j] == 0:
					conf_array[3] += 1
				else:
					conf_array[0] += 1
		conf.append(conf_array)
	return np.array(conf)


def evaluation_metric(y_act, y_pred):
	"""
	Determines precision and recall

	Parameters:
	==========
	y_act: 2D array, binarized actual labels
	y_pred: 2D array, binarized predicted labels

	Prints:
	=======
	Micro and macro precision and recall
	"""

	conf_mtx = get_confMatrix(y_act, y_pred)
	precision, recall = 0, 0
	for i in range(conf_mtx.shape[0]):
		TP, FP, TN, FN = conf_mtx[i]
		precision += (TP/(TP+FP) if (TP+FP) > 0 else 0)
		recall += (TP/(TP+FN) if (TP+FN) > 0 else 0)

	PrecisionMacro = precision / conf_mtx.shape[0]
	RecallMacro = recall / conf_mtx.shape[0]
	TP, FP, TN, FN = conf_mtx[:,0], conf_mtx[:,1], conf_mtx[:,2], conf_mtx[:,3]
	PrecisionMicro = np.sum(TP)/(np.sum(TP)+np.sum(FP))
	RecallMicro = np.sum(TP)/(np.sum(TP)+np.sum(FN))
	print("PrecisionMicroAvg : ", PrecisionMicro)
	print("PrecisionMacroAvg : ", PrecisionMacro)
	print("RecallMicroAvg : ", RecallMicro)
	print("RecallMacroAvg : ", RecallMacro)



def evaluate_model(model, df, test_size=0.2):
	"""
	Fits the train data on the given model and prints the results of evaluation metrics
	Parameters;
	==========
	model: Sklearn estimator to be used for fitting the data
	df: dataframe containing 100 features and 174 labels, (n_samples, 274)
	test_size: [default: 0.2] fraction of data to be taken as test data

	Returns: 
	========
	model: trained model
	"""
	
	train, test = train_test_split(df, test_size=test_size, random_state=42)
	train_id = train.iloc[:,0]
	test_id = test.iloc[:,0]

	X_train  = train.iloc[:,1:101]
	y_train = train.iloc[:,101:]
	X_test  = test.iloc[:,1:101]
	y_test = test.iloc[:,101:]

	y_pred = model.predict(X_test)

	evaluation_metric(np.array(y_test), y_pred)

	return model


def main():
	op_file = 'Model/LSVC_rpclassifier.model'
	inp_file = 'datasets/OneHotEnc.csv'
	df = pd.read_csv(inp_file)

	m = pickle.load(open(op_file, "rb"))
	model = evaluate_model(m, df)

	pickle.dump(model, open(op_file, 'wb'))



if __name__ == '__main__':
	main()
