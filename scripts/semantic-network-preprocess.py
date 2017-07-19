#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:57:50 2017

@author: emg
"""

import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/tidy-data/td_sample_comments_2017_05.csv')
subset = df[-df.author.isin(['AutoModerator'])]

def tokenizer(text):
    text = re.sub('[^A-Za-z0-9]+', ' ', text).strip(' ')
    text = re.findall('[a-zA-Z]{3,}', text)
    tokens = [word.lower() for word in text]
    return tokens

stopwords = stopwords.words('english')
stopwords.extend(['www','http','https','com','','html'])

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords,
                                    lowercase=True,
                                    tokenizer = tokenizer,
                                    #token_pattern=r'\b\w+\b',
                                     max_df=0.7, min_df=100)

corpus = [text for text in df['body']]
B = bigram_vectorizer.fit_transform(corpus)
B.toarray().shape
         
bigram_vectorizer.vocabulary_.get('reddit')

vocab_dict = bigram_vectorizer.vocabulary_
inv_map = {v: k for k, v in vocab_dict.items()}

m = pd.DataFrame(B.toarray()).rename(columns=inv_map)
m.shape
freq = m.sum().sort_values(ascending=False)
freq

cm = m.T.dot(m)

cm.to_csv('/Users/emg/Programming/GitHub/sub-text-analysis/tidy-data/td_word_co-matrix.csv')

import nltk
tags = nltk.pos_tag(words)

pos = list(list(zip(*tags))[1])
pos.sort()

from itertools import groupby
pos_freq = {}
for key, group in groupby(pos):
    pos_freq[key] = len(list(group)) 
pos_freq
