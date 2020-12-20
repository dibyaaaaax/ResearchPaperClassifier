# Research_Paper_Classifier

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Methodology](#methodology)
  * [Training](#training)
  * [Evaluation Metrics](#evaluation-metrics)
* [Getting Started](#getting-started)
  * [Installation](#installation)
  * [Reproduce Results and Visualizaton](#Reproduce Results and Visualizaton)



<!-- ABOUT THE PROJECT -->
## About The Project

An automated system which can do tagging of research papers. It gives multiple labels to a research paper classifier using the title and abstract of the research paper.


### Built With

* [Python](https://www.python.org/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)


## Methodology

The initial dataset has 1.7 million research papers in JSON format. The dataset was obtained from [Kaggle](https://www.kaggle.com/Cornell-University/arxiv). It has the research papers hosted on [arXiv](https://arxiv.org/).
The standard text preprocessing steps were applied to each of the data instances:
Lower case, remove stop words, punctuations and digits.

The following analysis were performed:
Number of data in each category. Number of categories for each data. Word cloud analysis for some classes.
Fasttext was used to learn the 100 dimensional word embeddings on the entire dataset of 1.7 million articles. Vocabulary of 30000 most frequent words were retained.
Feature for each document (Abstract + Title) was calculated as the average of all the words in the document.

### Training
The problem statement was decomposed into multiple independent classification problems. 
By using one vs rest classifier and logistic regression, for each label, a model was trained and tested on unseen data. Various evaluation metrics were analyzed. Eg: Micro and Macro Precision

The final model was trained using LinearSVC, and One-vs-rest method for multilabel classification. This model gave a better results than our baseline model. So, we tuned the hyperparameters and saved this model as our final model.

### Evaluation Metrics
Logistic Regression
```sh
Macro Precision : 0.88
Micro Precision : 0.96
Macro Recall : 0.48
Micro Recall : 0.45
```

Linear SVC
```sh
Macro Precision : 0.95
Micro Precision : 0.96
Macro Recall : 0.51
Micro Recall : 0.59
```
MLP Classifier
```sh
Macro Precision : 0.97
Micro Precision : 0.97
Macro Recall : 0.50
Micro Recall : 0.58
```
Random Forest Classifier
```sh
Macro Precision : 0.83
Micro Precision : 0.82
Macro Recall : 0.15
Micro Recall : 0.14
```

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.


### Installation

1. Clone the repository.
```sh
git clone https://github.com/nandinibansal2000/Research_Paper_Classifier.git
```
2. Run the following command to install all the required packages.
```sh
pip install -r requirements.txt 
```
3.  Run the following command to run the code on localhost.
```sh
python3 __init__.py
```

### Reproduce Results and Visualizaton
1. Download required files.
```sh
https://drive.google.com/file/d/1DobmfurhPQg6b1zKLyGAN2f28YouAn3F/view?usp=sharing
https://drive.google.com/file/d/1U93T9f42TwjCzyEPYd48yfltn50CnNae/view?usp=sharing
```
2. Run
```sh
python3 Clasifier/visualizations.py
python3 Classifier/evaluate_model.py
```


