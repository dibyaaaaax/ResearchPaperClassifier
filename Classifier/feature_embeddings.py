from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict
import json
import sys


def sampled_to_features(file, keyed_vec, verbose=True):
	"""
	convert text to features
	Parameters:
	==========
	file: json object containing all the sample, each with content and categories
	keyed_vec: vector file containing feature vector for all the words
	verbose: [default: True] print progress report is set

	Returns:
	=======
	dictionary containg features and labels
	"""

	sampled_data = defaultdict(list)

	for i, category in enumerate(file):
		if i%100000 == 0 and verbose:
			print(".", end=" ")

		# take the average of all vectors correponding to each word in the text
		sum_arr = np.zeros(keyed_vec.vector_size)
		total_words = 0
		for dic in file[category]:
			content = '</s> ' + dic['content']
			content = content.strip("\n").split(" ")

			for cont in content:
				if cont in keyed_vec:
					total_words += 1
					sum_arr += keyed_vec[cont]

			# add features and categories to a dictionary
			sampled_data["categories"].append(dic['categories'])    
			sampled_data["feature_vector"].append(sum_arr/total_words)
			
	return sampled_data


def learn_features(inp_file, op_file, vec_file):
	"""
	Parameters:
	===========
	inp_file: path to the file contaiing text data in pickled json form
	op_data: path to store the file containing feature vectors
	vec_file: file containing list of vectors for each word
	"""
	
	keyed_vec = KeyedVectors.load(vec_file)
	in_file = pickle.load(open(inp_file,"rb"))
	sampled_data = sampled_to_features(in_file,keyed_vec)

	with open(op_file, 'wb') as handle:
		pickle.dump(sampled_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



def main():
	op_file = sys.argv[1]
	inp_file = sys.argv[2]
	vec_file = sys.argv[3]
	learn_features(inp_file, op_file, vec_file)


if __name__ == '__main__':
	main()