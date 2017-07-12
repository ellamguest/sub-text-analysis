    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 15:36:26 2017

@author: emg
"""
import pandas as pd
import nltk
import re
from gensim import corpora, models
import matplotlib.pyplot as plt
import scipy as sp
from process_text import *

# import data and update manage variables
def prep_df():
    #df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/raw-data/td_comments_2017_05.csv')
    df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/raw-data/td_full_comments_2017_05.csv')
    df['time'] = pd.to_datetime(df['created_utc'], unit='s')
    df.sort_values('time', inplace=True)
    df['rank'] = df.groupby('author')['time'].rank()
    df['text_len'] = df['body'].map(lambda x:len(str(x)))
    df['author_count'] = df['author'].map(
            df.groupby('author').count()['time'])
    df['author_avg_score'] = df['author'].map(
            df.groupby('author').mean()['score'])
    df['active'] = df.author_count.apply(lambda x: 1 if x > 10 else 0)
    
    mods = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/td/master.csv')
    df['mod']=df.author.isin(mods['name'].unique()).map({False:0,True:1})
    df['tokens']=df['body'].apply(lambda x: stopless_stems(x))
    return df

df = prep_df()
df.to_csv('/Users/emg/Programming/GitHub/sub-text-analysis/tidy-data/td_full_comments_2017_05.csv')

# create subset by removing inappropraite authors
subset = df[df.author != '[deleted]']
subset = subset[subset.author != 'AutoModerator']

def author_count_score_plot(df, group):
    plt.scatter(x=df['score'], y=df['author_count'], c=df[group])
    plt.xlabel('comments score'), plt.ylabel('# comments by author')
    plt.title('Comment scores by author comment count (y={})'.format(group))
    plt.ylim(ymin=1)

author_count_score_plot(subset, 'mod')

def rank_score_plot(df, group):
    plt.scatter(x=df['rank'], y=df['score'], c=df[group])
    plt.xlabel('comment rank by author'), plt.ylabel('comment score')
    plt.title('Comment scores by rank in author comments (y={})'.format(group))
    plt.ylim(ymin=1)

rank_score_plot(subset, 'active')

authors = subset.drop_duplicates('author')[['author','author_count','active','mod','author_avg_score']]

##### print stats
print('There are {} authorless comments'.format(len(df[df['author']=='[deleted]'])))
print('There are {} comments by AutoModerator'.format(len(df[df['author']=='AutoModerator'])))
print('Comments by AutoModerator or without an author have been removed')
print('There are {} unique authors'.format(len(subset['author'].unique())))
print('{} authors made one comment'.format(len(subset[subset['author_count']==1])))
print('There are {} moderators among the authors'.format(len(set(subset[subset['mod']==1]['author']))))


#### comment histograms
subset.drop_duplicates('author')
subset[subset['author_count'] > 10].drop_duplicates('author').author_count.hist()

count = subset.groupby('author_count').count()['count']
comment_counts = pd.DataFrame({'count':count.index,'freq':count,'relfreq':count/16000})
comment_counts['cumfreq'] = comment_counts.relfreq.cumsum()
comment_counts.loc[0] = [0,0,0,0]
comment_counts.sort_index(inplace=True)
comment_counts.plot(x='count',y='cumfreq',
                    title='Author Comment # Cumulative Freq')

plt.plot(comment_counts['count'], comment_counts['cumfreq'])
plt.xlabel('# comments by author'), plt.ylabel('% total comments')
plt.title('Author Comment # Cumulative Freq')
plt.xlim(xmin=0, xmax=comment_counts.index[-1]), plt.ylim(ymin=0)

#### scores
stats = df['score'].describe()
print('The scores range from {} to {}'.format(stats['min'], stats['max']))
print('The interquartile range is {} to {}'.format(stats['25%'], stats['75%']))
print('The median is {} and the mode is {}'.format(df.score.median(), df.score.mode()[0]))

print('The lowest scoring comment is: \n')
print(df.iloc[-1]['body'])

print('The highest scoring comment is: \n')
print(df.iloc[0]['body'])


repeats = subset[subset.groupby('author').author.transform(len) > 1]

print('There appears to be no correlation between author comment frequency and score')

subset.plot(y='author_count',x='score',kind='scatter')


print('There appears to be no correlation between comment length and score')
repeats.plot('text_len','score', kind='scatter')

### looking at most prolific authors
print('There appears to be no improvement in scores by author over time')
top = count[2:9].index
for name in top:
    subset = repeats[repeats['author']==name]
    subset.plot(x='rank',y='score',kind='scatter', title=name)
    
  
 



### desc stats by group
def comment_score_stats_table(df):
    desc_stats = pd.DataFrame({'total':df.score.describe()})
    desc_stats['not_deleted'] = df[df['author']!='[deleted]'].score.describe()
    desc_stats['deleted'] = df[df['author']=='[deleted]'].score.describe()
    desc_stats['mods'] = df[df['mod']==1].score.describe()
    desc_stats['non_mods'] = df[-df['mod']==0].score.describe()
    desc_stats['active(>10)'] = df[df['active']==1].score.describe()
    desc_stats['inactive'] = df[df['active']==0].score.describe()
    stats = desc_stats.T
    stats['prop'] = stats['count']/df.shape[0]
    return stats
stats = comment_score_stats_table(subset)
stats


stats.to_csv('score_by_group.csv')



#### desc stats by author type


texts = [text for text in df.head(100)['body']]

from sklearn import feature_extraction, externals, cluster, metrics, manifold
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.05, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(texts) #fit the vectorizer to sidebar revisions
terms = tfidf_vectorizer.get_feature_names()



# word analysis
texts = list(topdf['body'])

stopwords = nltk.corpus.stopwords.words('english')
def tokenize(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for word in nltk.word_tokenize(text)]
    stopless_tokens = [word for word in tokens if word not in stopwords]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in stopless_tokens:
        if re.match('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

stemmer = nltk.stem.snowball.SnowballStemmer("english")
def stem(tokens): return [stemmer.stem(token) for token in tokens]

all_tokens = []
all_stems = []
for text in texts:
    current_tokens = tokenize(text)
    all_tokens.extend(current_tokens)
    
    current_stems = stem(current_tokens)
    all_stems.extend(current_stems)
    
vocab_frame = pd.DataFrame({'word':all_tokens, 'stem':all_stems})

print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print('there are ' + str(len(set(vocab_frame['stem']))) + ' stems in vocab_frame')


count = vocab_frame.groupby('stem').count()
count.sort_values('word',inplace=True, ascending=False)
count['rel_freq'] = count['word'] / len(count)
count = count[count['words']>29]

freq = count.groupby('word').count()
freq.sort_values('rel_freq',inplace=True, ascending=False)
topstems = count[count['word']>29]
topstems



''' TOPIC MODELLING - Latent Dirichlet Allocation
LDA is a probabilistic topic model that assumes documents are a mixture of
topics and that each word in the document is attributable to the document's topics
EXAMPLE REMOVED PROPER NOUNS, I HAVE NOT
'''

'''text is a list of sidebar revisions returns from prep_texts'''
#tokenized_text = [tokenize_and_stem(text) for text in texts] #tokenize
tokenized_text = [tokenize(text) for text in active['body']] #tokenize
stopless_texts = [[word for word in text if word not in stopwords] for text in tokenized_text] #remove stop words


texts = [text for text in df['tokens'].head()]                
dictionary = corpora.Dictionary(texts) #create a Gensim dictionary from the texts
dictionary.filter_extremes(no_below=20) #remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
corpus = [dictionary.doc2bow(text) for text in tokens] #convert the dictionary to a bag of words corpus for reference

# run the model, set number of topics
lda = models.LdaModel(corpus, num_topics=3, id2word=dictionary, update_every=5, chunksize=10000, passes=100)

# convert the topics into just a list of the top 20 words in each topic
topics_matrix = lda.show_topics(formatted=False, num_words=20)

for n in range(len(topics_matrix)):
    topic_words = topics_matrix[n][1]
    print('TOPIC {}'.format(n))
    print()
    print([str(word[0]) for word in topic_words])
    print()
    
    
#tfidf

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]
lsi.print_topics(2)

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
corpus_lda = lda[corpus_tfidf]
lda.print_topics(2)
