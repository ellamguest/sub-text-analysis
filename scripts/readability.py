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
    text = re.sub(r"&gt;", "", text).strip() # common bug should be >
    text = re.sub(r"http\S+", "", text) #remove all urls
    text = re.sub(r"(haha)\S+", "", text) #remove any number of hahas
    text = re.sub(' +',' ', text)
    #text = re.sub('[^\w. ]','', text.lower())
    #text = re.sub('[^A-Za-z0-9]+', ' ', text).strip(' ')
    if re.search('[a-zA-Z]', text) == None:
        text = ''
    return text


exclude = list(string.punctuation)
def counts(text):
        #ignoreCount = 0
    sentences = re.split(r'\n| *[\.\?!][\'"\)\]]* *', text)
    n_sentences = len([s for s in sentences if s not in [None, '']])
#    for sentence in sentences:
#        if n_words <= 2: # try to parameterize!
#            ignoreCount = ignoreCount + 1
    #n_sentences =  max(1, len(sentences) - ignoreCount)
    
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
    plt.scatter(x=df[x], y=df[y])
    plt.xlabel(x), plt.ylabel(y)
    if col != None:
        print('{}'.format(col))
        plt.title('{} {} by {} (colour = {})'.format(sub, x,y, col))
    else:
        print('no color')
        plt.title('{} {} by {}'.format(sub, x,y)) 
    
    
def colour_plot(sub, df, x, y, col=None):
    plt.scatter(x=df[x], y=df[y], c=df[col])
    plt.xlabel(x), plt.ylabel(y)
    if col=None
    plt.title('{} {} by {} (colour = {})'.format(sub, x,y, col))

def double_plot(df1, df2, x, y):
    plt.scatter(x=df1[x], y=df1[y], c='blue', alpha=0.25)
    plt.scatter(x=df2[x], y=df2[y], c='red', alpha=0.25)
    plt.xlabel(x), plt.ylabel(y)
    plt.title('{} by {}'.format(x,y))


td_df = basic_df('td')
td_read = get_readability_measures(td_df)
td = add_difficult_words(td_read)

td_outliers = [1522]


cmv_df = basic_df('cmv')
cmv_read = get_readability_measures(cmv_df)
cmv = add_difficult_words(cmv_read)

test = td_read.drop(1522)


plot('td', td, 'FRE', 'FKG', col='ASL') # trying to combine plot but not working
colour_plot('td', td, 'FRE', 'FKG', col='ASL')

plot(cmv_read, 'FRE', 'FKG', '{} comment'.format('cmv'))

colour_plot('TD', td, 'ASL', 'ASW', 'DIFFW')

## difficult word frequencies
from itertools import groupby

            
x = td_read['DIFFW_SET'].iloc[0]

## trying to get a way of measuring common terms
def diff_word_freq(df):
    words = []
    for difficult_words in df['DIFFW_SET']:
        words.extend(difficult_words)
    words.sort()
    freq = [(len(list(group)), key) for key, group in groupby(words)]
    freq.sort()
    
    rev_freq = {}
    for key, value in freq:
        rev_freq[value] = key
    return rev_freq

def diff_word_count(diff_word_set, rev_freq):
    n = 0
    for word in diff_word_set:
        n += rev_freq[word]
    return n

rev_freq = diff_word_freq(cmv_df)
cmv_read['diff_weighted_count'] = cmv_read['DIFFW_SET'].apply(lambda x: diff_word_count(x, rev_freq))


plot(cmv_read, 'diff_weighted_count', 'FKG', 'cmv comment')
plot(cmv_read, 'DIFFW', 'FKG', 'cmv comment')

