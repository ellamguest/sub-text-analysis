#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:10:30 2017

@author: emg
"""

import re

text =  "&gt; This is my test text. \n I'm seeing hahahahaha! hahahaha &gt; WHAT RE CAN DO%$^£^..// 195. http://blahblah.com asfh34"


re.split('[\w]\.',text)# how to get this to not drop letter before .
text = re.split('(?<=\w)[.!?]',text) # positive lookbehind assertion

         
        
         
def clean_text(text):
text = re.sub(r"&gt;", "", text).strip() # common bug should be >
text = re.sub(r"http\S+", "", text) #remove all urls
text = re.sub(r"(haha)+\S", "", text) #remove any number of hahas
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

text = re.sub("&gt;|http\S+|(haha)\S+|  +", "", text).strip() # common bug should be >
text = re.sub(' +',' ', text)
if re.search('[a-zA-Z]', text) == None:
    text = ''

re.split('\n|(?<=\w)[.!?]|\n',text) # split at end punctutation if preceded by alphanumeri
re.split(r'\n| *[\.\?!][\'"\)\]]* *', text)    

      
re.sub(r"(haha)+[\b ]", "", text) # removed anything betwene ha has to - must fix
