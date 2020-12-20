import sys
from gensim.models import KeyedVectors
import numpy as np
import pickle
import json
import arxiv

from LearnEmbeddings.feature_preprocess import Preprocess


def predict(data, vec_file, model, targets):
	"""
	Predicts a list of labels for a text composed of title and abstract
	Parameters:
	===========
	data: string to be classified
	vec_file: file that has features for each word
	model: sklearn model object that is to be used for prediction
	targets: dictionary containing a list of all the classes

	Returns:
	========
	classes: a list of tuple containing (predicted_class, probability)
	"""

	# Preprocess the "data" and get the feature vector for each word
	preprocessor = Preprocess()
	preprocessed_data = preprocessor.basic_cleanup(data)
	print(preprocessed_data, "\n")
	keyed_vec = KeyedVectors.load(vec_file)
	sum_arr = np.zeros(keyed_vec.vector_size)

	content = '</s> ' + data
	content = content.strip("\n").split(" ")
	total_words = 0

	# add up the feature vector for each word and take an average
	for cont in content:
		if cont in keyed_vec:
			total_words += 1
			sum_arr += keyed_vec[cont]

	features = np.array([sum_arr/total_words])
	print(features)

	#predict the labels and get the probabilities
	pred = model.predict(features)
	# pred_proba = model.predict_proba(features)
	# classes = [(targets[idx], pred_proba[0][idx]) for idx in range(len(pred[0])) if pred[0][idx] == 1]
	classes = [targets[idx] for idx in range(len(pred[0])) if pred[0][idx] == 1]


	return classes


def main():
	vec_file = sys.argv[1]
	model_file = sys.argv[2]

	# read the Arxiv Id for the paper to be classified and make an API call to Arxiv to get the data
	ids = input("Arxiv ID: ")
	doc = arxiv.query(id_list=[ids])
	title = doc[0]['title']
	abstract = doc[0]['summary']

	# load all the necessary files and make the predictin
	data = " ".join([title, abstract])
	model = pickle.load(open(model_file, 'rb'))
	targets = json.load(open("datasets/targets.json"))
	classes = predict(data, vec_file, model, targets['Labels'])
	print(classes)


if __name__ == '__main__':
	main()