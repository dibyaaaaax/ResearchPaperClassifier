import sys
import re
import json
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


class Preprocess:
    """
    Preprocesses the json file downoaded from Arxiv and makes it ready for Fasttext
    """
    def __init__(self, sv=False):
        """
        Parameters:
        sv: if set, the input to Fasttext will include the labels as well
        """

        self.supervised = sv
        pass


    def basic_cleanup(self, doc):
        """
        Perform basic preprocessing on the text data
        Parameters:
        =========
        doc: text to be cleaned up

        returns;
        =======
        preprocessed text
        """

        doc = doc.lower().strip()
        doc = doc.replace("-", " ")
        doc = re.sub(r'\d+', '', doc)
        doc = doc.translate(str.maketrans("","", string.punctuation))

        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(doc)
        result = [i for i in tokens if not i in stop_words]
        return " ".join(result)
    
  

    def preprocess(self, fileName, verbose):
        """
        Cleans up the required text and make it ready for the input to fasttext. 
        This data wil be used by fastttext to learn the word embeddings.

        Parameters:
        fileName: path to the json file from Arxiv
        verbose: print the progress while processing the file
        """

        file = open(fileName)
        for i, p in enumerate(file):
            if i%10000 == 0 and verbose:
                print(".", end="")
            paper = json.loads(p)
            doc = " ".join([paper['title'], paper['abstract']])
            doc = self.basic_cleanup(doc)
            
            if self.supervised:
                _categories = paper['categories'].split(" ")
                categories = ["__label__" + cat for cat in _categories]
                cat_str = " ".join(categories)
                doc = cat_str + " " + doc

            yield doc



    def preprocessFile(self, inputFile, outputFile, verbose=True):
        """
        call the preprocess function and store the results to appropriate locations

        Parameter:
        =========
        inputFile: path to the input file
        outputFile: path to the output file
        verbose: [default: True] prints progress if set
        """
        
        op_file = open(outputFile, 'w')
        for doc in self.preprocess(inputFile, verbose):
            op_file.write(doc + "\n")
        op_file.close()
        

def main():
    pp = Preprocess()
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    pp.preprocessFile(inputFile, outputFile)

if __name__ == '__main__':
    main()

