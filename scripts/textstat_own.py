#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:03:29 2017

@author: emg
"""

#from __future__ import print_function
#import pkg_resources
import string
import re
#import math
#import operator

exclude = list(string.punctuation)
easy_word_set = [line.rstrip() for line in open('/Users/emg/Programming/GitHub/sub-text-analysis/resources/easy_words.txt', 'r')]       

def char_count(text, ignore_spaces=True):
    """
    Function to return total character counts in a text, pass the following parameter
    ignore_spaces = False
    to ignore whitespaces
    """
    if ignore_spaces:
        text = text.replace(" ", "")
    return len(text)

def lexicon_count(text, removepunct=True):
    """
    Function to return total lexicon (words in lay terms) counts in a text
    """
    if removepunct:
        text = ''.join(ch for ch in text if ch not in exclude)
    count = len(text.split())
    return count

def syllable_count(text):
    """
    Function to calculate syllable words in a text.
    I/P - a text
    O/P - number of syllable words
    """
    count = 0
    vowels = 'aeiouy'
    text = text.lower()
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
        count = count - (0.1*count) # why syllables 0.9 each not 1?
        return count

def sentence_count(text):
    """
    Sentence count of a text
    """
    ignoreCount = 0
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
    for sentence in sentences:
        if lexicon_count(sentence) <= 2:
            ignoreCount = ignoreCount + 1
    return max(1, len(sentences) - ignoreCount)

def avg_sentence_length(text):
    lc = lexicon_count(text)
    sc = sentence_count(text)
    try:
        float(lc/sc)
        return round(lc/sc, 1)
    except:
        print("Error(ASL): Sentence Count is Zero, Cannot Divide")
        return

def avg_syllables_per_word(text):
    syllable = syllable_count(text)
    words = lexicon_count(text)
    try:
        ASPW = float(syllable)/float(words)
        return round(ASPW, 1)
    except:
        print("Error(ASyPW): Number of words are zero, cannot divide")
        return

def avg_letter_per_word(text):
    try:
        ALPW = float(float(char_count(text))/float(lexicon_count(text)))
        return round(ALPW, 2)
    except:
        print("Error(ALPW): Number of words are zero, cannot divide")
        return

def avg_sentence_per_word(text):
    try:
        ASPW = float(float(sentence_count(text))/float(lexicon_count(text)))
        return round(ASPW, 2)
    except:
        print("Error(AStPW): Number of words are zero, cannot divide")
        return

def flesch_reading_ease(text):
    ASL = avg_sentence_length(text)
    ASW = avg_syllables_per_word(text)
    FRE = 206.835 - float(1.015 * ASL) - float(84.6 * ASW)
    return round(FRE, 2)

def flesch_kincaid_grade(text):
    ASL = avg_sentence_length(text)
    ASW = avg_syllables_per_word(text)
    FKRA = float(0.39 * ASL) + float(11.8 * ASW) - 15.59
    return round(FKRA, 1)

def polysyllabcount(text):
    count = 0
    for word in text.split():
        wrds = syllable_count(word)
        if wrds >= 3:
            count += 1
    return count

def difficult_words(text):
    text_list = text.split()
    diff_words_set = set()
    for value in text_list:
        if value not in easy_word_set:
            if syllable_count(value) > 1:
                if value not in diff_words_set:
                    diff_words_set.add(value)
    return len(diff_words_set)

def difficult_words_set(text):
    text_list = text.split()
    diff_words_set = set()
    for value in text_list:
        if value not in easy_word_set:
            if syllable_count(value) > 1:
                if value not in diff_words_set:
                    diff_words_set.add(value)
    return diff_words_set

