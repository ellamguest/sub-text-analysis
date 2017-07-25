#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:36:35 2017

@author: emg
"""

import pandas as pd
import re
from semantic-network-preprocess import check_rule, sentiment_variables, tokenizer
from textstat.textstat import textstat

sub = 'cmv'

def prep_df(sub):
    df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/raw-data/{}_sample_comments_2017_06.csv'.format(sub))
    df = df[-df.author.isin(['AutoModerator', 'DeltaBot', '[deleted]'])]
    df = df[-df.body.isin(['[deleted]','', '∆'])]
    mods = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/master.csv'.format(sub))
    df['time'] = pd.to_datetime(df['created_utc'], unit='s')
    df = df.assign(
            rule_comment = df['body'].apply(lambda x: check_rule(x)),
            rank= lambda df: df.groupby('author')['time'].rank(),
            text_len= lambda df: df['body'].apply(lambda x:len(str(x))),
            author_count= lambda df: df['author'].map(
            df.groupby('author').count()['time']),
            mod = lambda df:df.author.isin(mods['name'].unique()).map({False:0,True:1}),
            author_avg_score= lambda df: df['author'].map(
            df.groupby('author').mean()['score']))
    df = df[df['rule_comment'] == False]
    df = sentiment_variables(df)
    df['tokens'] = df['body'].apply(lambda x: tokenizer(x))
    df['token_length'] = df['tokens'].apply(lambda x: len(x))
    return df

def simple_text(text):
    text = re.sub(r"http\S+", "", text) #remove complete urls
    text = re.sub('[^A-Za-z0-9]+', ' ', text).strip(' ')
    text = re.findall('[a-zA-Z]{3,}', text)
    tokens = [word.lower() for word in text]
    simple_text = ' '.join(tokens)
    return simple_text

def clean_text(df):
    texts = df[df['token_length']>0]['body']
    texts = [re.sub(' +',' ', text) for text in texts]
    texts = [re.sub(r"&gt;", "", text).strip() for text in texts] # common bug should be >
    texts = [re.sub(r"http\S+", "", text) for text in texts]
    texts = [re.sub('[^\w. ]','', text) for text in texts]
    texts = list(filter(None, texts))
    return texts
    
def readability_df(df):
    texts = clean_text(df)
    
    DIFFW = [textstat.difficult_words(text) for text in texts]
    FKG = [textstat.flesch_kincaid_grade(text) for text in texts]
    FRE = [textstat.flesch_reading_ease(text) for text in texts]
    
    readability_df = pd.DataFrame({'text':texts,'DIFFW':DIFFW,
                                   'FKG':FKG,'FRE':FRE,})
    
    readability_df['FRE_level'] = pd.cut(FRE, 
                  bins = [-1000, 0, 30, 50, 60, 70, 80, 90, 100, 1000],
                  labels=['above college grad', 'college grad', 
                          'college student', '10-12th grade',
                           '8-9th grade', '7th grade', 
                           '6th grade', '5th grade',
                           'lower than 5th grade'])
    return readability_df

df = prep_df('td')
read = readability_df(df)

read[read['FRE']>100]

plot(read, 'FRE', 'FKG', '{} comment'.format(sub))
plot(read, 'FRE', 'DIFFW', '{} comment'.format(sub))

td_readability = readability_df(td)

'''
DIFFW =  number of difficult words
FKG = flesch-kincaid grade level
FRE = flesch reading ease #can be negative, affect by polysyllabic words
SMOG = Simple Measure Of Gobbledygook #comments too short for SMOG

FRE matrix
100.00-90.00	5th grade	Very easy to read. Easily understood by an average 11-year-old student.
90.0–80.0	6th grade	Easy to read. Conversational English for consumers.
80.0–70.0	7th grade	Fairly easy to read.
70.0–60.0	8th & 9th grade	Plain English. Easily understood by 13- to 15-year-old students.
60.0–50.0	10th to 12th grade	Fairly difficult to read.
50.0–30.0	College	Difficult to read.
30.0–0.0	College Graduate
'''
text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

def avg_syllables_per_word(self, text):
        syllable = self.syllable_count(text)
        words = self.lexicon_count(text)
        try:
            ASPW = float(syllable)/float(words)
            return round(ASPW, 1)
        except:
            print("Error(ASyPW): Number of words are zero, cannot divide")
            return

