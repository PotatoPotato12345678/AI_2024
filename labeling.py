import collections
import pathlib
import random
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

ast = open('./processed/astronomy.txt', 'r').read().split("\n\n")
psy_1 = open('./processed/psychology.txt', 'r').read().split("\n\n")
psy_2 = open('./processed/psychology_emotions.txt', 'r').read().split("\n\n")
scio_1 = open('./processed/sociology.txt', 'r').read().split("\n\n")
scio_2 = open('./processed/sociology_added.txt', 'r').read().split("\n\n")

each_data_num = 2000
corpus = [ast[:each_data_num], (psy_1+psy_2)[:each_data_num], (scio_1 + scio_2)[:each_data_num]]

idx = 0
data_label = {}
for field_idx in range(len(corpus)):
    for abstract in corpus[field_idx]:
        data_label[idx] = {
            "label": field_idx,
            "abstract": abstract
            }   
        
        idx+=1

with open('pickle/data.pickle', mode='wb') as fo:
  pickle.dump(data_label,fo)