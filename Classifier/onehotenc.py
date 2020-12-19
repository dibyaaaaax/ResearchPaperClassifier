import pickle
from pandas import DataFrame
import pandas as pd
import numpy as np
import sys
import json


def onehotenc(inp_file, op_file, target):
	"""
	binarizes the labels and stores the data as a dataframe containing
	100 features and 174 binarized labels

	Parameters:
	==========
	inp_file: json containg list of features and list of labels
	op_file: path where the op dataframe is to be saved
	target: python dictionary containing list of all the labels
	"""
	veclen = len(inp_file["feature_vector"][0])
	labels_list = target['Labels']
	features_list = ["feature_" + str(i) for i in range(1, veclen+1)]
	df = []

	for f in range(len(inp_file["categories"])):
		label = inp_file["categories"][f]
		content = inp_file["feature_vector"][f]
		label_idx = [0]*len(labels_list)
		for l in label:
			idx = labels_list.index(l)
			label_idx[idx] = 1
		temp = np.concatenate((content, label_idx),axis = 0)
		df.append(temp)

	df = DataFrame(df,columns = features_list+labels_list)
	df.to_csv(op_file, index = False)


def main():
	op_file = sys.argv[1]
	inp_file = sys.argv[2]
	file = open(inp_file, "rb")
	inp_file = pickle.load(file)
	targets = json.load(open("datasets/targets.json"))
	onehotenc(inp_file, op_file, targets)


if __name__ == '__main__':
	main()

