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
    subset = subset[-subset.body.isin(['[deleted]','', '∆'])]
    subset = subset[subset['rule_comment'] == False]
    subset = subset[subset['rule_comment'] == False]
    return subset

############## prep df

def get_df(sub):
    df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/raw-data/{}_sample_comments_2017_06.csv'.format(sub))
    mods = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/master.csv'.format(sub))
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
    return df


td = get_df('td')
cmv = get_df('cmv')

td = subset_comments(td)
cmv = subset_comments(cmv)

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

m, freq, cm = ngrams_by_comment(subset, ngram_range=(1,1), min_df=100)

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
def plot(df, x, y, unit_name):
    plt.scatter(x=df[x], y=df[y])
    plt.xlabel(x), plt.ylabel(y)
    plt.title('{} {} by {}'.format(unit_name, x,y))

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


####### slangsd

slangsd = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/SlangSD/SlangSD.csv', sep='\t', names=['slang','sentiment'])
s = pd.Series

slangsd['sent'] = slangsd.sentiment.apply(lambda x: x/2)
slang_dict = slangsd[['slang','sent']].set_index('slang').to_dict()

s = pd.Series(data=slangsd['sent'])
s.index = slangsd['slang']
s.to_dict()

x = pd.DataFrame({'word':freq.index})
x['sentiment'] = x['word'].apply(lambda x: sentiment(x))
x['polarity'] = x['sentiment'].apply(lambda x:x[0])
x['subjectivity'] = x['sentiment'].apply(lambda x:x[1])




########## comparing td and cmv
td = get_df('td')
cmv = get_df('cmv')

td = subset_comments(td)
cmv = subset_comments(cmv)


tdm, tdfreq, tdcm = ngrams_by_comment(td, ngram_range=(1,1), min_df=100)

cmvm, cmvfreq, cmvcm = ngrams_by_comment(cmv, ngram_range=(1,1), min_df=100)

def word_sentiment(freq):
    x = pd.DataFrame({'word':freq.index})
    x['sentiment'] = x['word'].apply(lambda x: sentiment(x))
    x['polarity'] = x['sentiment'].apply(lambda x:x[0])
    x['subjectivity'] = x['sentiment'].apply(lambda x:x[1])
    return x

tdsent = word_sentiment(tdfreq)
cmvsent = word_sentiment(cmvfreq)


tdsent.plot('polarity','subjectivity',kind='scatter')
cmvsent.plot('polarity','subjectivity',kind='scatter')

plot(tdsent, 'polarity','subjectivity', 'td word')
plot(cmvsent, 'polarity','subjectivity', 'cmv word')

overlap = []
for word in list(tdsent['word']):
    if word in list(cmvsent['word']):
        overlap.append(word)
        
td_only = []
for word in list(tdsent['word']):
    if word not in list(cmvsent['word']):
        td_only.append(word)
        
tdrelfreq = tdfreq.apply(lambda x: x/tdfreq.sum())
tdrelfreq.plot(title='td word rel freq plot')

cmvrelfreq = cmvfreq.apply(lambda x: x/cmvfreq.sum())
cmvrelfreq.plot(title='cmv word rel freq plot')

rf = []
for word in overlap:
    rf.append([tdrelfreq[word], cmvrelfreq[word]])

plt.scatter(list(zip(*rf))[0], list(zip(*rf))[1])
plt.xlabel('word rel freq in td')
plt.ylabel('word rel freq in cmv')
plt.title('relative word frequencies')
plt.xlim(xmin=0), plt.ylim(ymin=0)

for label, x, y in zip(overlap, list(zip(*rf))[0], list(zip(*rf))[1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

tdnums = tdsent.set_index('word')
tdnums['relfreq'] = tdrelfreq
      
cmvnums = cmvsent.set_index('word')
cmvnums['relfreq'] = cmvrelfreq

def plot(df, x, y, unit_name):
    plt.scatter(x=df[x], y=df[y])
    plt.xlabel(x), plt.ylabel(y)
    plt.title('{} {} by {}'.format(unit_name, x,y))  
    plt.ylim(ymin=0)    
       
plot(tdnums, 'polarity', 'relfreq', 'td words')
plot(cmvnums, 'polarity', 'relfreq', 'cmv words')
