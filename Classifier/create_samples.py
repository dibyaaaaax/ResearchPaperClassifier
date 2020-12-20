import sys
import pickle
from collections import defaultdict
import json
import random

from LearnEmbeddings.feature_preprocess import Preprocess


class CreateSample:
	"""
	Given a data, preprocesses it and creates a balanced data if required
	"""
	
	def __init__(self, sample_size=None):
		self.sample_size = sample_size


	def create_sample(self, imp_file, verbose=True):
		"""
		creates a balanced sample for each class and returns the sampled data

		Parameters:
		inp_file: path to input file containing imbalanced data
		verbose: [default: True] show the progress report

		Returns:
		sampled_data: python dictionart containing id, preprocessed content and categories of the balanced data
		"""
		print(self.sample_size)
		file = open(imp_file)
		sampled_data = defaultdict(list)
		pp = Preprocess()
		for i, p in enumerate(file):
			paper = json.loads(p)
			categories = paper['categories'].split(" ")
			dic = {
					'id': paper['id'],
					'content': pp.basic_cleanup(paper['title'] +" "+paper['abstract']),
					'categories': categories
				}
			for cat in categories:
				sampled_data[cat].append(dic)
				
			if i%100000 == 0 and verbose:
				print(".", end=" ")

		# if the sample size is given, create a balanced sample
		if not self.sample_size is None:
			for i in sampled_data:
				sampled_data[i] = random.sample(sampled_data[i], 
									min(len(sampled_data[i]), self.sample_size))

		return sampled_data


	def create_sample_to_file(self, inp_file, op_file):
		"""
		Create balanced data and store in a file

		Parameters:
		inp_file: path to input file containing imbalanced data
		op_file: path to the destination for the file contaioning balanced data
		"""

		sampled_data = self.create_sample(inp_file)
		with open(op_file, 'wb') as handle:
			pickle.dump(sampled_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
	op_file = sys.argv[1]
	inp_file = sys.argv[2]
	sample_size = int(sys.argv[3])
	cs = CreateSample(sample_size)
	cs.create_sample_to_file(inp_file, op_file)



if __name__ == '__main__':
	main()