#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:36:35 2017

@author: emg
"""

import pandas as pd
import re
import string
import numpy as np
import matplotlib.pyplot as plt
#from textstat_own import *

def check_rule(text):
    rules = ['&gt; Comment Rule', 'n&gt; Submission Rule',
             'Removed, see comment rule',
             'http://www.reddit.com/r/changemyview/wiki/rules'] 
    for rule in rules:
        if rule in text:
            return True
    else:
        return False
    
def basic_df(sub):
    df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/raw-data/{}_sample_comments_2017_06.csv'.format(sub))
    df = df[-df.author.isin(['AutoModerator', 'DeltaBot', '[deleted]'])]
    df = df[-df.body.isin(['[deleted]','', '∆'])]
    df['rule_comment'] = df['body'].apply(lambda x: check_rule(x))
    df = df[df['rule_comment'] == False]
    return df

def clean_text(text):
    text = re.sub("&gt;|http\S+|(haha)\S+|  +", "", text).strip() # common bug should be >
    text = re.sub(' +',' ', text)
    if re.search('[a-zA-Z]', text) == None:
        text = ''
    return text

def tokenizer(text):
    text = re.sub(r"http\S+", "", text) #remove urls
    text = re.sub('[^A-Za-z0-9]+', ' ', text).strip(' ')
    text = re.findall('[a-zA-Z]{3,}', text)
    tokens = [word.lower() for word in text]
    return tokens

exclude = list(string.punctuation)
def counts(text):
    sentences = re.split('\n|(?<=\w)[.!?]|\n',text) # split at end punctutation if preceded by alphanumeric
    n_sentences = len([s for s in sentences if s not in [None, '']])
    
    count = 0
    vowels = 'aeiouy'
    text = text.lower()
    text = "".join(ch for ch in text if ch not in exclude)
    n_words = len(text.split(' '))

    if text is None:
        count = 0
    elif len(text) == 0:
        count = 0
    else:
        if text[0] in vowels:
            count += 1
        for index in range(1, len(text)):
            if text[index] in vowels and text[index-1] not in vowels:
                count += 1
        if text.endswith('e'):
            count -= 1
        if text.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        count = count - (0.1*count) # why syllables 0.9 each not 1?
    n_syllables = count

    return n_syllables, n_words, n_sentences

easy_word_set = [line.rstrip() for line in open('/Users/emg/Programming/GitHub/sub-text-analysis/resources/easy_words.txt', 'r')]       
def difficult_words_set(text):
    text_list = text.split()
    diff_words_set = set()
    n_syllables = counts(text)[0]
    for value in text_list:
        if value not in easy_word_set:
            if n_syllables > 1:
                if value not in diff_words_set:
                    diff_words_set.add(value)
    return diff_words_set


def get_readability_measures(df):
    df['text'] = df['body'].apply(lambda x: clean_text(x))
    df = df.ix[-df['text'].isin([None, ''])].copy()
    df['tokens'] = df['text'].apply(lambda x: tokenizer(x))
    
    df['n_syllables'], df['n_words'], df['n_sentences'] = list(zip(*
                      df['text'].apply(lambda x: counts(x))))
    
    df['ASL'] = np.divide(df['n_words'], df['n_sentences']) #gives error when dividing by 0
    df['ASW'] = np.divide(df['n_syllables'], df['n_words'])
    
    df['FRE'] = 206.835 - (float(1.015) * df['ASL']) - (float(84.6) * df['ASW'])
    df['FKG'] = (float(0.39) * df['ASL']) + (float(11.8) * df['ASW']) - 15.59
      
    return df

def add_difficult_words(df):
    '''separted from get_readibilty_measures because takes a while'''
    print('Getting difficult words sets....')
    df['DIFFW_SET'] = df['text'].apply(lambda x: difficult_words_set(x))
    print('Done with difficult words sets....')
    df['DIFFW'] = df['DIFFW_SET'].apply(lambda x: len(x))
    df['rel_diffw'] = np.divide(df['n_words'],df['DIFFW'])
    
    return df
    
def plot(sub, df, x, y, col=None):
    if col==None:
        plt.scatter(x=df[x], y=df[y])
        plt.xlabel(x), plt.ylabel(y)
        plt.title('{} {} by {}'.format(sub, x,y)) 
    else:
        sc = plt.scatter(x=df[x], y=df[y], c=df[col])
        plt.colorbar(sc)
        plt.xlabel(x), plt.ylabel(y)
        plt.title('{} {} by {} (colour = {})'.format(sub, x,y, col))


def double_plot(df1, df2, x, y, cols=['blue','red']):
    plt.scatter(x=df1[x], y=df1[y], c=cols[0], alpha=0.25)
    plt.scatter(x=df2[x], y=df2[y], c=cols[1], alpha=0.25)
    plt.xlabel(x), plt.ylabel(y)
    plt.title('{} by {}'.format(x,y))


df = basic_df('td').head(10)
td_df = basic_df('td')
td_read = get_readability_measures(td_df)
td = add_difficult_words(td_read)

plot('TD', td_read, 'FRE', 'FKG')
plot('TD', td_read, 'FRE', 'FKG', col='ASW') # trying to combine plot but not working

td_outliers = [1522] # comments with particularly non-standard syntax/punctuation

cmv_df = basic_df('cmv').head(50)
cmv_read = get_readability_measures(cmv_df)
cmv = add_difficult_words(cmv_read)


## difficult word frequencies
from itertools import groupby
        
x = td_read['DIFFW_SET'].iloc[0]

## trying to get a way of measuring common terms
def word_freq(word_series):
    all_words = []
    for words in word_series:
        all_words.extend(words)
    all_words.sort()
    freq = [(len(list(group)), key) for key, group in groupby(all_words)]
    freq.sort()
    
    rev_freq = {}
    for key, value in freq:
        rev_freq[value] = key
    return rev_freq

def freq_word_count(word_set, rev_freq):
    n = 0
    for word in word_set:
        n += rev_freq[word]
    return n

rev_freq = word_freq(td_read['tokens'])
td_read['weighted_count'] = td_read['tokens'].apply(lambda x: freq_word_count(x, rev_freq))

td['diff_weighted_count'] = td['DIFFW_SET'].apply(lambda x: freq_word_count(x, rev_freq))

plot('TD', df, 'weighted_count', 'score')
plot('TD', td, 'DIFFW', 'FKG')
plot('TD', td_read, 'weighted_count', 'score', 'n_words')

