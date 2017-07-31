#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:36:35 2017

@author: emg
"""

import pandas as pd
import re
from textstat.textstat import textstat
import matplotlib.pyplot as plt
import string
import pkg_resources
import math
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

def difficult_words_set(text):
        text_list = text.split()
        diff_words_set = set()
        for value in text_list:
            if value not in easy_word_set:
                if textstat.syllable_count(value) > 1:
                    if value not in diff_words_set:
                        diff_words_set.add(value)
        return diff_words_set

def readability_df(df):
    texts = df['body'].apply(lambda x: clean_text(x))
    texts = list(filter(None, texts))
    
    DIFFW = [difficult_words(text) for text in texts]
    DIFFW_SET = [difficult_words_set(text) for text in texts]
    FKG = [flesch_kincaid_grade(text) for text in texts]
    FRE = [flesch_reading_ease(text) for text in texts]
    
    readability_df = pd.DataFrame({'body':df['body'],'text':texts,
                                   'DIFFW':DIFFW, 'DIFFW_SET':DIFFW_SET,
                                   'FKG':FKG,'FRE':FRE,})
    
    readability_df['FRE_level'] = pd.cut(FRE, 
                  bins = [-2000, 0, 30, 50, 60, 70, 80, 90, 100, 2000],
                  labels=['above college grad', 'college grad', 
                          'college student', '10-12th grade',
                           '8-9th grade', '7th grade', 
                           '6th grade', '5th grade',
                           'lower than 5th grade'])
    readability_df['num_words'] = readability_df['text'].apply(lambda x: len(x.split(' ')))
    readability_df['rel_diffw'] = readability_df['num_words']/readability_df['DIFFW']
    readability_df['rel_diffw'].replace(np.inf, np.nan, inplace=True)
    
    return readability_df

def plot(df, x, y, unit_name):
    plt.scatter(x=df[x], y=df[y])
    plt.xlabel(x), plt.ylabel(y)
    plt.title('{} {} by {}'.format(unit_name, x,y)) 


sub = 'td'
df = basic_df(sub)
df = df.head(20)
read = readability_df(df)
td_read = read
long = read.ix[read['num_words']>=10]
short = read.ix[read['num_words']<10]

short.ix[short['FKG']<0][['text','FKG','num_words']]



plot(read, 'FRE', 'FKG', '{} comment'.format(sub))
plot(read, 'FRE', 'rel_diffw', '{} comment'.format(sub))
plot(read, 'num_words', 'DIFFW', '{} comment'.format(sub))
plot(short, 'num_words', 'FKG', '{} short comments'.format(sub))




sub = 'cmv'
df = basic_df(sub)
read = readability_df(df)


plot(read, 'FRE', 'FKG', '{} comment'.format(sub))
plot(read, 'FRE', 'DIFFW', '{} comment'.format(sub))
plot(read, 'FRE', 'rel_diffw', '{} comment'.format(sub))
plot(read, 'num_words', 'DIFFW', '{} comment'.format(sub))
plot(read, 'DIFFW_new', 'DIFFW', '{} comment'.format(sub))

read['rel_diffw'].replace(np.inf, np.nan, inplace=True)
read['log_rel_diffw'] = read['rel_diffw'].apply(lambda x: math.log(x))
read['log_diffw'] = read['DIFFW_new'].apply(lambda x: math.log(x))
plot(read, 'FRE','log_rel_diffw', sub)


####### deconstructing DIFFW
def syllable_count(self, text):
        """
        Function to calculate syllable words in a text.
        I/P - a text
        O/P - number of syllable words
        """
        count = 0
        vowels = 'aeiouy'
        text = text.lower()
        exclude = list(string.punctuation)
        text = "".join(x for x in text if x not in exclude)

        if text is None:
            return 0
        elif len(text) == 0:
            return 0
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
            count = count - (0.1*count)
            return count
        
easy_word_set = [line.rstrip() for line in open('/Users/emg/Programming/GitHub/sub-text-analysis/resources/easy_words.txt', 'r')]       


easy_word_set = set([ln.strip().decode('utf-8') for ln in pkg_resources.resource_stream('textstat', 'easy_words.txt')])
        
def difficult_words_list(text):
        text_list = text.split()
        diff_words_set = set()
        for value in text_list:
            if value not in easy_word_set:
                if textstat.syllable_count(value) > 1:
                    if value not in diff_words_set:
                        diff_words_set.add(value)
        return diff_words_set

difficult_words_list(text)
len(difficult_words_list(text)) == textstat.difficult_words(text)
textstat.difficult_words(text)

