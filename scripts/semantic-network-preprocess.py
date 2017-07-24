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

#m, freq, cm = ngrams_by_comment(subset, ngram_range=(1,1), min_df=100)

#cm.to_csv('/Users/emg/Programming/GitHub/sub-text-analysis/tidy-data/cmv_word_co-matrix.csv')


############# to select for pos types

#def pos_tags(m):
#    tags = nltk.pos_tag(m.columns)
#    
#    pos = list(list(zip(*tags))[1])
#    pos.sort()
#    
#    pos_freq = {}
#    for key, group in groupby(pos):
#        pos_freq[key] = len(list(group)) 
#    
#    return tags, pos, pos_freq
#
#tags, pos, pos_freq = pos_tags(m)
#pos_freq

############ plots
def plot(df, x, y, unit_name):
    plt.scatter(x=df[x], y=df[y])
    plt.xlabel(x), plt.ylabel(y)
    plt.title('{} {} by {}'.format(unit_name, x,y))

#plot(subset, 'polarity', 'score')



############# basic stats
def stats_table(df):
    stats = []
    for col in list(df.columns):
        stats.extend(df[col].describe().index)
        
    table = pd.DataFrame(index=set(stats), columns=df.columns)
    for col in df.columns:
        table[col] = df[col].describe()
    return table


####### slangsd

#slangsd = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/SlangSD/SlangSD.csv', sep='\t', names=['slang','sentiment'])
#s = pd.Series
#
#slangsd['sent'] = slangsd.sentiment.apply(lambda x: x/2)
#slang_dict = slangsd[['slang','sent']].set_index('slang').to_dict()
#
#s = pd.Series(data=slangsd['sent'])
#s.index = slangsd['slang']
#s.to_dict()
#
#x = pd.DataFrame({'word':freq.index})
#x['sentiment'] = x['word'].apply(lambda x: sentiment(x))
#x['polarity'] = x['sentiment'].apply(lambda x:x[0])
#x['subjectivity'] = x['sentiment'].apply(lambda x:x[1])
#



########## comparing td and cmv
td = get_df('td')
cmv = get_df('cmv')

td = subset_comments(td)
cmv = subset_comments(cmv)

tdm, tdfreq, tdcm = ngrams_by_comment(td, ngram_range=(1,1), min_df=100)

cmvm, cmvfreq, cmvcm = ngrams_by_comment(cmv, ngram_range=(1,1), min_df=100)

def word_df(freq_series):
    df = pd.DataFrame({'word':freq_series.index, 'freq':freq_series})
    df['sentiment'] = df['word'].apply(lambda x: sentiment(x))
    df['polarity'] = df['sentiment'].apply(lambda x:x[0])
    df['subjectivity'] = df['sentiment'].apply(lambda x:x[1])
    df['relfreq'] = df['freq'].apply(lambda x: x/freq_series.sum())
    return df

tdwords, cmvwords = word_df(tdfreq), word_df(cmvfreq)

overlap = tdwords.merge(cmvwords, how='inner', on='word')[['word',
                       'freq_x','relfreq_x', 'freq_y','relfreq_y',  
                       'polarity_y', 'subjectivity_y']]
overlap.columns = ['word','freq_td','relfreq_td', 'freq_cmv','relfreq_cmv',
                   'polarity', 'subjectivity']
        
td_only = []
for word in list(tdsent['word']):
    if word not in list(cmvsent['word']):
        td_only.append(word)




######## plots  
def plot(df, x, y, unit_name, x_min=False, y_min=False):
    plt.scatter(x=df[x], y=df[y])
    plt.xlabel(x), plt.ylabel(y)
    plt.title('{} {} by {}'.format(unit_name, x,y))  
    if x_min==True:
        plt.xlim(xmin=0) 
    if y_min==True:
        plt.ylim(ymin=0) 
      

plot(overlap, 'relfreq_td','relfreq_cmv', 'word', x_min=True, y_min=True)
common = overlap[overlap['relfreq_td']>0.011]
for label, x, y in zip(common.word, common.relfreq_td, common.relfreq_cmv):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))


plot(tdwords, 'polarity','subjectivity', 'td word')
plot(cmvwords, 'polarity','subjectivity', 'cmv word')   
       
plot(tdnums, 'polarity', 'relfreq', 'td words')
plot(cmvnums, 'polarity', 'relfreq', 'cmv words')

tdnums['word'] = tdnums.index
tdnums['word_length'] = tdnums['word'].apply(lambda x: len(x))

cmvnums['word'] = cmvnums.index
cmvnums['word_length'] = cmvnums['word'].apply(lambda x: len(x))

plt.hist(tdnums['word_length'])
plt.title('Hist of td word lengths')