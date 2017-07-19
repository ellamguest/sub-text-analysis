#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:05:41 2017

@author: emg
"""

import boto3
import botocore

session = boto3.Session()
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
    
bucket = s3.Bucket('emg-td-comments')

BUCKET_NAME = 'emg-td-comments'
KEY = 'td_full_comments_2017_05.csv'

try:
    s3.Bucket(BUCKET_NAME).download_file(KEY, 'my_local_image.jpg')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise
        
        
        
client = boto3.client('ec2')





client = boto3.client('s3', 'us-west-2')
transfer = S3Transfer(client)

transfer.upload_file('/tmp/myfile', BUCKET-NAME, 'RC_2017-06.xz')

new_file = 'https://files.pushshift.io/reddit/comments/RC_2017-06.xz'