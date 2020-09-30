from gensim.models import KeyedVectors
import sys


def vec_to_gensim(input_path, output_path, limit=30000):
    model = KeyedVectors.load_word2vec_format(input_path)
    model.save(output_path)


def main():
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    limit = int(sys.argv[3])
    vec_to_gensim(inputFile, outputFile, limit)



if __name__ == '__main__':
    main()