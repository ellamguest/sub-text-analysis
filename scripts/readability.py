#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:36:35 2017

@author: emg
"""

import pandas as pd
import re
import matplotlib.pyplot as plt
#import string
#import pkg_resources
#import math
import numpy as np
from textstat_own import *

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
    text = re.sub(r"#\S+", "", text) #removing all urls - lack of spaces messes w/ readabilty scores 
    text = re.sub(r"(haha)\S+", "", text.lower()) #remove any number of hahas
    text = re.sub(' +',' ', text)
    text = re.sub('[^\w. ]','', text)
    text = re.sub('[^A-Za-z0-9]+', ' ', text).strip(' ')
    return text


def get_readability_measures(df):
    df['text'] = df['body'].apply(lambda x: clean_text(x))
    df = df.ix[-df['text'].isin([None, ''])].copy()
    
    df['DIFFW_SET'] = df['text'].apply(lambda x: difficult_words_set(x))
    df['DIFFW'] = df['DIFFW_SET'].apply(lambda x: len(x))
    df['FKG'] = df['text'].apply(lambda x: flesch_kincaid_grade(x))
    df['FRE'] = df['text'].apply(lambda x: flesch_reading_ease(x))
    df['ASL'] = df['text'].apply(lambda x: avg_sentence_length(x))
    df['ASW'] = df['text'].apply(lambda x: avg_syllables_per_word(x))
    df['num_words'] = df['text'].apply(lambda x: len(x.split(' ')))
    df['rel_diffw'] = df['num_words']/df['DIFFW']

    return df

def plot(df, x, y, unit_name):
    plt.scatter(x=df[x], y=df[y])
    plt.xlabel(x), plt.ylabel(y)
    plt.title('{} {} by {}'.format(unit_name, x,y)) 
    
def colour_plot(df, x, y, unit_name, col_col):
    plt.scatter(x=df[x], y=df[y], c=df[col_col])
    plt.xlabel(x), plt.ylabel(y)
    plt.title('{} {} by {}'.format(unit_name, x,y))


df = basic_df('td')
td_read = get_readability_measures(df)

df = basic_df('cmv')
cmv_df = df
cmv_read = get_readability_measures(cmv_df)



plot(td_read, 'FRE', 'FKG', 'td comment')
plot(td_read, 'FRE', 'DIFFW', 'td comment')
plot(td_read, 'FRE', 'rel_diffw', 'td comment')
plot(td_read, 'num_words', 'DIFFW', 'td comment')


colour_plot(td_read, 'FRE', 'FKG', '{} comment - colour = ASL'.format('td'), col_col='ASL')
colour_plot(td_read, 'FRE', 'FKG', '{} comment - colour = ASW'.format('td'), col_col='ASW')

colour_plot(td_read, 'ASL', 'ASW', '{} comment - colour = FRE'.format('td'), col_col='FRE')
colour_plot(td_read, 'ASL', 'ASW', '{} comment - colour = FKG'.format('td'), col_col='FKG')

td_scored = td_read.ix[td_read['score']!=1].copy()
plot(td_scored, 'num_words', 'score', 'td comment')
plot(td_scored, 'FKG', 'score', 'td comment')
plot(td_scored, 'FRE', 'score', 'td comment')

colour_plot(td_read, 'ASL', 'ASW', 'td comment - colour = score', col_col='score')


plot(cmv_read, 'FRE', 'FKG', '{} comment'.format('cmv'))
plot(cmv_read, 'FRE', 'DIFFW', '{} comment'.format('cmv'))
plot(cmv_read, 'FRE', 'rel_diffw', '{} comment'.format('cmv'))
plot(cmv_read, 'num_words', 'DIFFW', '{} comment'.format('cmv'))

colour_plot(cmv_read, 'FRE', 'FKG', '{} comment - colour = ASL'.format('cmv'), col_col='ASL')
colour_plot(cmv_read, 'FRE', 'FKG', '{} comment - colour = ASW'.format('cmv'), col_col='ASW')

colour_plot(cmv_read, 'ASL', 'ASW', '{} comment - colour = FRE'.format('cmv'), col_col='FRE')
colour_plot(cmv_read, 'ASL', 'ASW', '{} comment - colour = FKG'.format('cmv'), col_col='FKG')

cmv_scored = cmv_read.ix[cmv_read['score']!=1].copy()
plot(cmv_scored, 'num_words', 'score', 'cmv comment')
plot(cmv_scored, 'FKG', 'score', 'cmv comment')
plot(cmv_scored, 'FRE', 'score', 'cmv comment')

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

