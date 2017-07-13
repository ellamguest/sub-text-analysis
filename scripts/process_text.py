#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:55:18 2017

@author: emg
"""

import re
from nltk.corpus import stopwords
from nltk.stem import snowball
import pandas as pd
   
stemmer = snowball.SnowballStemmer("english")
stop = stopwords.words('english')
                      
def stopless_stems(text):
    text = re.sub('[^A-Za-z0-9]+', ' ', text).strip(' ')
    tokens = [word.lower() for word in text.split(' ')]
    contractionless = [term for term in tokens if term not in contractions]
    stems = [stemmer.stem(t) for t in contractionless]
    stopless = tuple(term for term in stems if term not in stop)
    return stopless

def prep_df():
    #df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/raw-data/td_comments_2017_05.csv')
    df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/raw-data/td_full_comments_2017_05.csv')
    df['time'] = pd.to_datetime(df['created_utc'], unit='s')
    #df.sort_values('time', inplace=True)
    df['rank'] = df.groupby('author')['time'].rank()
    df['text_len'] = df['body'].map(lambda x:len(str(x)))
    df['author_count'] = df['author'].map(
            df.groupby('author').count()['time'])
    df['author_avg_score'] = df['author'].map(
            df.groupby('author').mean()['score'])
    df['active'] = df.author_count.apply(lambda x: 1 if x > 10 else 0)
    
    mods = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/td/master.csv')
    df['mod']=df.author.isin(mods['name'].unique()).map({False:0,True:1})
    #df['tokens']=df['body'].apply(lambda x: stopless_stems(x))
    return df

df = prep_df()
df.to_csv('/Users/emg/Programming/GitHub/sub-text-analysis/tidy-data/td_full_comments_2017_05.csv')


contractions = { 
"ain't": "am not; are not; is not; has not; have not",
"aren't": "are not; am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}
