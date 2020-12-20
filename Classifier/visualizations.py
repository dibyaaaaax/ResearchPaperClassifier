import pandas as pd
import numpy as np
import os
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import rand
from IPython.display import HTML
from matplotlib import animation

'''PCA'''

df = pd.read_csv("/datasets/OneHotEnc.csv")
df.head()

pd.options.display.max_columns = None
pd.options.display.max_rows = None

train, test = train_test_split(df, test_size = 0.2)
X  = test.iloc[:,:100]
y = test.iloc[:,100:]
categories = list(y.columns.values)

pca = PCA(n_components=3, svd_solver='arpack')
X_pca = pca.fit_transform(X)

y['new'] = y.apply(lambda x: x.index[x == 1].tolist(), axis=1)
y['new'].shape

c=0
plot_X = []
plot_y = []
for i, y_i in enumerate(y['new']):
  X_i = X_pca[i]
  if len(y_i) == 1:
    y_i = y_i[0].split(".")[0]
    plot_X.append(X_i)
    plot_y.append(y_i)

colormap = {'physics':'tab:blue',
            'quant-ph':'tab:red',
            'math':'tab:purple', 
            'gr-qc':'tab:brown',
            'econ':'tab:pink',
            'q-fin':'tab:gray',
            'astro-ph':'tab:olive',
            'q-bio':'tab:orange',
            'nucl-th':'tab:green',
            'hep-ex':'tab:cyan',
            'nucl-ex':'khaki',
            'cs':'springgreen',
            'hep-ph':'darkred',
            'stat':'violet',
            'cond-mat':'lightpink',
            'eess':'indigo',
            'nlin':'black',
            'hep-th':'tan',
            'hep-lat': 'lightblue'
            }

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
for i in range(len(plot_X)): # plot each point + it's index as text above
  _x = plot_X[i][0]
  _y = plot_X[i][1]
  _z = plot_X[i][2]

  label = plot_y[i]

  ax.scatter(_x, _y, _z, color=colormap[label], s=50)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

c=0
plot_cs_X = []
plot_cs_y = []
for i, y_i in enumerate(y['new']):
  X_i = X_pca[i]
  if len(y_i) == 1 and y_i[0].startswith('astro-ph'):
    plot_cs_X.append(X_i)
    plot_cs_y.append(y_i[0])

colormap1 = {'astro-ph.CO':'tab:blue',
            'astro-ph':'tab:red',
            'astro-ph.HE':'tab:purple', 
            'astro-ph.SR':'tab:brown',
            'astro-ph.GA':'tab:pink',
            'astro-ph.EP':'tab:gray',
            'astro-ph.IM':'tab:olive'
            }

fig1 = plt.figure(figsize=(20,20))
ax1 = fig1.add_subplot(111, projection='3d')

for i in range(len(plot_cs_X)): # plot each point + it's index as text above
  _x = plot_cs_X[i][0]
  _y = plot_cs_X[i][1]
  _z = plot_cs_X[i][2]

  label = plot_cs_y[i]

  ax1.scatter(_x, _y, _z, color=colormap1[label], s=50)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')



FILE_PATH = "datasets/arxiv-metadata-oai-snapshot.json"
file = open(FILE_PATH)

# Creating an imbalanced sample

def create_imbalanced_sample(file):
    sampled_data = defaultdict(list)
    num_classes = defaultdict(int)
    for i, p in enumerate(file):
        paper = json.loads(p)
        categories = paper['categories'].split(" ")
        dic = {
                'id': paper['id'],
                'title': paper['title'],
                'abstract':paper['abstract'],
                'categories': categories
            }
        for cat in categories:
            sampled_data[cat].append(dic)

        num_classes[len(categories)] += 1
            
        if (i%100000 == 0):
            print(".", end=" ")
            
    return sampled_data, num_classes


file = open(FILE_PATH)
imbalanced_sample, num_classes = create_imbalanced_sample(file)

class_counts = []
for cat in imbalanced_sample.keys():
  class_counts.append((cat, len(imbalanced_sample[cat])))

df = pd.DataFrame(class_counts, columns =['Name', 'number']) 
num_classes_arr = [(i, j) for i, j in num_classes.items()]
df_num_classes = pd.DataFrame(num_classes_arr, columns=['num_tags', 'num_datapoints'])
pd.set_option('display.max_rows', None)
df = df.sort_values('number')

targets = list(df["Name"])

# Plotting the number of data samples in each class

fig = plt.figure() 
ax = fig.add_subplot()
ax.tick_params(axis='y', which='major', labelsize=20)
ax.tick_params(axis='x', which='major', labelsize=10)
df.plot.bar(x='Name', y='number', title="Number in each class", figsize=(25,45), ax=ax)

# Plotting the number of papers with given number of tags

df_num_classes.plot.bar(x='num_tags', y='num_datapoints', title="Number in each class", figsize=(15,5), color='green');

