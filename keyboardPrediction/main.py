import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

import streamlit as st

"""
## Tokenize by regex
"""
path = '1661-0.txt'
text = open(path).read().lower()
st.write('len: ', len(text))


"""
## Tokenize by regex
"""

tokenizer = RegexpTokenizer(r'\w+')
words =  tokenizer.tokenize(text)


"""
## Unique words index
"""

unique_words = np.unique(words)
unique_word_index = dict((c, i) for i,c in enumerate(unique_words))

"""
## Feature: Next 5 words
"""
WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])
st.write(prev_words[0])
st.write(next_words[0])

"""
## Feature: Next 5 words
"""


X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)))
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)

for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        x[i,j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1

st.write(X[0][0])
