from gensim.models import KeyedVectors
import sys


def vec_to_gensim(input_path, output_path, limit=30000):
	"""
	Convert the output from fasttext to gensim format which is easy to read
	Parameters:
	input_path: path to the input file
	output_path: path to the output gensim file
	limit: number of most frequent words to keep. 
	"""
    model = KeyedVectors.load_word2vec_format(input_path)
    model.save(output_path)


def main():
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    limit = int(sys.argv[3])
    vec_to_gensim(inputFile, outputFile, limit)



if __name__ == '__main__':
    main()