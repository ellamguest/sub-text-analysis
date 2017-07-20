#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:57:50 2017

@author: emg
"""

import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from itertools import groupby

sample = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/tidy-data/td_sample_comments_2017_05.csv')
#df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/tidy-data/td_full_comments_2017_05.csv')
cmv = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/raw-data/cmv_sample_comments_2017_06.csv')

def tokenizer(text):
    text = re.sub('[^A-Za-z0-9]+', ' ', text).strip(' ')
    text = re.findall('[a-zA-Z]{3,}', text)
    tokens = [word.lower() for word in text]
    return tokens

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['www','http','https','com','','html'])

def ngrams_by_comment(df, ngram_range=(1, 2), min_df=100):
    subset = df[-df.author.isin(['AutoModerator', 'DeltaBot', '[deleted]'])]
    corpus = [text for text in subset['body'].dropna()]
    print('There are {} comments in the corpus'.format(len(corpus)))
    bigram_vectorizer = CountVectorizer(ngram_range = ngram_range, stop_words=stopwords,
                                        lowercase=True,
                                        tokenizer = tokenizer,
                                        min_df = min_df)
    
    B = bigram_vectorizer.fit_transform(corpus)
         
    vocab_dict = bigram_vectorizer.vocabulary_
    inv_map = {v: k for k, v in vocab_dict.items()}
    
    m = pd.DataFrame(B.toarray()).rename(columns=inv_map)
    print('There are {} ngrams in the corpus'.format(m.shape[1]))
    
    freq = m.sum().sort_values(ascending=False)
    cm = m.T.dot(m)
    
    return m, freq, cm

m, freq, cm = ngrams_by_comment(cmv, ngram_range=(2,4), min_df=50)

cm.to_csv('/Users/emg/Programming/GitHub/sub-text-analysis/tidy-data/cmv_word_co-matrix.csv')

def pos_tags(m):
    tags = nltk.pos_tag(m.columns)
    
    pos = list(list(zip(*tags))[1])
    pos.sort()
    
    pos_freq = {}
    for key, group in groupby(pos):
        pos_freq[key] = len(list(group)) 
    
    return tags, pos, pos_freq

tags, pos, pos_freq = pos_tags(m)
pos_freq

