import sys
import re
import json
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')


class Preprocess:
    def __init__(self, sv=False):
        self.supervised = sv
        pass


    def basic_cleanup(self, doc):
        doc = doc.lower().strip()
        doc = doc.replace("-", " ")
        doc = re.sub(r'\d+', '', doc)
        doc = doc.translate(str.maketrans("","", string.punctuation))

        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(doc)
        result = [i for i in tokens if not i in stop_words]
        return " ".join(result)
    
  

    def preprocess(self, fileName, verbose):
        file = open(fileName)
        for i, p in enumerate(file):
            if i==100:
              break
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

