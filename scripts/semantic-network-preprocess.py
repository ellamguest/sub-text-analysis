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
from textblob import TextBlob
import matplotlib.pyplot as plt

#### td sample
#sample = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/tidy-data/td_sample_comments_2017_05.csv')

#### td full
#df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/tidy-data/td_full_comments_2017_05.csv')

######### text pre-processing functions

def check_rule(text):
    rules = ['&gt; Comment Rule', 'n&gt; Submission Rule',
             'Removed, see comment rule',
             'http://www.reddit.com/r/changemyview/wiki/rules'] 
    for rule in rules:
        if rule in text:
            return True
    else:
        return False
    
def tokenizer(text):
    text = re.sub('[^A-Za-z0-9]+', ' ', text).strip(' ')
    text = re.findall('[a-zA-Z]{3,}', text)
    tokens = [word.lower() for word in text]
    return tokens

def sentiment(text):
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    return pol,subj

def sentiment_variables(df):
    df['sentiment'] = df['body'].apply(lambda x: sentiment(x))
    df['polarity'] = df['sentiment'].apply(lambda x: x[0])
    df['subjectivity'] = df['sentiment'].apply(lambda x: x[1])
    return df

def subset_comments(df):
    subset = df[-df.author.isin(['AutoModerator', 'DeltaBot', '[deleted]'])]
    subset = subset[-subset.body.isin(['[deleted]',''])]
    subset = subset[subset['rule_comment'] == False]
    return subset

############## prep df
df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/raw-data/cmv_sample_comments_2017_06.csv')
mods = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/cmv/master.csv')
df['time'] = pd.to_datetime(df['created_utc'], unit='s')
df = df.assign(
        rule_comment = lambda df: df['body'].pipe(check_rule),
        rank= lambda df: df.groupby('author')['time'].rank(),
        text_len= lambda df: df['body'].apply(lambda x:len(str(x))),
        author_count= lambda df: df['author'].map(
        df.groupby('author').count()['time']),
        mod = lambda df:df.author.isin(mods['name'].unique()).map({False:0,True:1}),
        author_avg_score= lambda df: df['author'].map(
        df.groupby('author').mean()['score']))

df = sentiment_variables(df)

subset = subset_comments(df)

############## get breakdown n grams
###### m = document x n gram matrix
###### freq = frew dist of n grams
###### cm = n gram collocation matrix

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['www','http','https','com','','html'])

def ngrams_by_comment(subset, ngram_range=(1, 2), min_df=100):
    corpus = [text for text in subset['body'].dropna() if text is not '∆']
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

m, freq, cm = ngrams_by_comment(subset, ngram_range=(2,3), min_df=50)

cm.to_csv('/Users/emg/Programming/GitHub/sub-text-analysis/tidy-data/cmv_word_co-matrix.csv')

############# to select for pos types
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







############ plots
def plot(df, x, y):
    plt.scatter(x=df[x], y=df[y])
    plt.xlabel(x), plt.ylabel(y)
    plt.title('comment {} by {}'.format(x,y))

plot(subset, 'polarity', 'score')
plot(subset, 'polarity','subjectivity')
plot(subset, 'text_len','score')

non1 = subset[subset['score'] != 1]
plot(non1, 'text_len','score')


############# basic stats
def stats_table(df):
    stats = []
    for col in list(df.columns):
        stats.extend(df[col].describe().index)
        
    table = pd.DataFrame(index=set(stats), columns=df.columns)
    for col in df.columns:
        table[col] = df[col].describe()
    return table

table = stats_table(subset)
table


####### troubleshooting rule commnet removal
def check_rule(text):
    rules = ['&gt; Comment Rule', 'n&gt; Submission Rule',
             'Removed, see comment rule',
             'http://www.reddit.com/r/changemyview/wiki/rules'] 
    for rule in rules:
        if rule in text:
            return True
    else:
        return False

n = 1
for text in df['body']:
    if check_rule(text) == True:
        print(text)
        n+=1
    

errors = []
test = m[m['reddit changemyview wiki']==1]['reddit changemyview wiki']
for i in test.index:
    print(i)
    errors.append(subset.iloc[i]['body'])
    
rule_test = []
for error in errors:
    rule_test.append(check_rule(error))
