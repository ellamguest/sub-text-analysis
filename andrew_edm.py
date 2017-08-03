#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:25:47 2017

@author: emg
"""
import re
import pandas as pd
import numpy as np

def xml2df(xml_path):
    xml_data = open(xml_path).read()
    text = xml_data
    text = re.sub('</id>|</session>|</number>|</title>|</text>|</proposer>|</signature_count>|</signature>|</date>|</motion>|</mp>|</type>', '', text) #start could
    text = re.split("[\n][\t]*", text)
    ids = [item.lstrip('<id>') for item in text if item.startswith('<id>') == True]
    session_date = [item.lstrip('<session>') for item in text if item.startswith('<session>') == True]
    session_number = [item.lstrip('<number>') for item in text if item.startswith('<number>') == True]
    title = [item.lstrip('<title>') for item in text if item.startswith('<title>') == True]
    body = [item.lstrip('<text>') for item in text if item.startswith('<text>') == True]
    proposer = [item.lstrip('<proposer id=">') for item in text if item.startswith('<proposer id=') == True]
    signature_count = [item.lstrip('<signature_count>') for item in text if item.startswith('<signature_count>') == True]
    
    mp_id = [item.lstrip('<mp id="') for item in text if item.startswith('<mp id=') == True]
    sign_date = [item.lstrip('<date>') for item in text if item.startswith('<date>') == True]
    sign_type = [item.lstrip('<type>') for item in text if item.startswith('<type>') == True]
    
    motion_df = pd.DataFrame({'id':ids,'session_date':session_date,
                       'session_number':session_number, 'title':title,
                       'body':body, 'proposer_id':proposer,'signature_count':signature_count})
    motion_df['signature_count'] = motion_df['signature_count'].astype(int) #4710 signatures
        
    df = motion_df.loc[np.repeat(motion_df.index.values,motion_df.signature_count)].copy()
    df['mp_id'] = mp_id
    df['signature_date'] = sign_date
    df['signature_type'] = sign_type
    df['proposer_id'], df['proposer_name'] = list(zip(*df['proposer_id'].str.split('">')))
    df['mp_id'], df['mp_name'] = list(zip(*df['mp_id'].str.split('">')))
   # df['last_name'], df['first_name'] = list(zip(*df['mp_name'].str.split(', ')))
    
    return df

xml_paths = ['/Users/emg/Desktop/edms/2010-12.xml',
            '/Users/emg/Desktop/edms/2012-13.xml',
            '/Users/emg/Desktop/edms/2013-14.xml',
            '/Users/emg/Desktop/edms/2014-15.xml',
            '/Users/emg/Desktop/edms/2015-16.xml',
            '/Users/emg/Desktop/edms/2016-17.xml',
            '/Users/emg/Desktop/edms/2017-18.xml']

dfs = []
for xml_path in xml_paths:
    df = xml2df(xml_path)
    dfs.append(df)
total = pd.concat(dfs)
total.to_csv('/Users/emg/Desktop/edms/edm-signatures-2010-2017.csv')


import xml.etree.ElementTree as etree
from lxml import etree
parser = etree.XMLParser(recover=True)
tree = etree.fromstring('/Users/emg/Desktop/edms/2016-17.xml', parser=parser)
root = tree.getroot()  

from xml.etree.ElementTree import ElementTree
from xml.parsers import expat
tree = ElementTree()
xml_file = xml_paths[-1]
root = tree.parse(xml_file, parser=expat.ParserCreate('UTF-8') )
root = tree.parse(xml_file, parser=expat.ParserCreate('UTF-16') )
root = tree.parse(xml_file, parser=expat.ParserCreate('ISO-8859-1') )
root = tree.parse(xml_file, parser=expat.ParserCreate('ASCII') )


<?xml version="1.0" encoding="UTF-8"?>








#########33 jewish mps
from bs4 import BeautifulSoup
def make_soup(url):
    '''makes soup object from url using requests module '''
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html5lib")
    return soup

# 9692 EDM ids
# 3218 session numbers

jmp_html = '/Users/emg/Desktop/edms/List of British Jewish politicians - Wikipedia.html'
html_data = open(jmp_html).read()

soup = BeautifulSoup(html_data)

# h2 where id = British_MPs
