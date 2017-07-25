#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:36:35 2017

@author: emg
"""

from textstat.textstat import textstat
 
texts = td[td['token_length']>0]['body']

SMOG = texts.apply(lambda x: textstat.smog_index(x))

FKG = texts.apply(lambda x: textstat.flesch_kincaid_grade(x))

FRE = texts.apply(lambda x: textstat.flesch_reading_ease(x))

DIFFW = texts.apply(lambda x: textstat.difficult_words(x))

df = pd.DataFrame({'text':texts,'SMOG':SMOG,'FKG':FKG,'FRE':FRE,'DIFFW':DIFFW})


n = 0
for text in texts:
    print(n)
    try:
        textstat.flesch_kincaid_grade(text)
    except TypeError:
        0
    n+=1

lambda:  for _ in ()).throw(Exception(0))

def fkg(text):    
    try:
        fkg = textstat.flesch_kincaid_grade(text)
    except TypeError:
        fkg = 0  
    return fkg
    
texts.apply(lambda x: fkg(x))
