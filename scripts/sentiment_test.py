#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:07:40 2017

@author: emg
"""

import math
from textblob import TextBlob as tb
import pandas as pd
from basics import prep_df
import nltk

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

bloblist = [text for text in df.head(100)['body']]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
        
from nltk.sentiment import SentimentAnalyzer
sid = SentimentAnalyzer()
for sentence in bloblist:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
        print()
        
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([doc for doc in bloblist])

tokens = df['tokens'][2]
tokens
tagged = nltk.pos_tag(tokens)
tagged

