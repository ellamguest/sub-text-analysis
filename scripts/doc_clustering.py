# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 18:01:01 2017

@author: emg
"""
# using code from http://brandonrose.org/clustering
import numpy as np
import pandas as pd
import nltk
import re
from sklearn import feature_extraction, externals, cluster, metrics, manifold
import scipy.cluster.hierarchy as hca
from gensim import corpora, models, similarities 


##############################################################################
##############################################################################

''' PREPPING DATA '''

df = pd.read_csv('/Users/emg/Programming/GitHub/sub-text-analysis/raw-data/td_comments_2017_03.csv')
df.head()

df['time'] = pd.to_datetime(df['created_utc'], unit='s')
df.sort_values('time', inplace=True)


sample = df.sample(300)

texts = [text for text in df['body']]

##############################################################################
##############################################################################

''' CLEANING DATA (Stopwords, stemming, and tokenizing) '''

stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.snowball.SnowballStemmer("english")

def tokenize_and_stem(text):
      # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens1 = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    tokens = []
    for word in tokens1:
        if word not in stopwords:
            tokens.append(word)
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    tokens = [word for word in tokens if word not in stopwords]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens
    
# create two lists: 1) all words tokenized + stemmed, 2) all words tokenized only
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in texts:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

# create a vocab df with index = stems and columns = all matching tokens
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print('there are ' + str(len(set(vocab_frame.index))) + ' stems in vocab_frame')



vocab_frame['stem'] = vocab_frame.index
count = vocab_frame.groupby('stem').count()
count.sort_values('words',inplace=True, ascending=False)
count['rel_freq'] = count['words'] / len(count)
count = count[count['words']>29]









##############################################################################
##############################################################################

texts = [text for text in df['body']]

''' TF-IDF DOCUMENT SIMILARITY

 step 1) count word occurrences by document.
 step 2) create a document-term matrix (dtm) aka term frequency matrix
 step 3) apply the term frequency-inverse document frequency weighting
    
 parameter notes:
    max_df: if the term is in greater than 80% of the documents it probably cares little meanining , play with
    min_idf: this could be an integer (e.g. 5) and the term would have to be in at least 5 of the documents to be considered, play with 
    ngram_range: set length on n-grams considered
'''

#define vectorizer parameters
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.05, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(texts) #fit the vectorizer to sidebar revisions
terms = tfidf_vectorizer.get_feature_names() #a list of the features used in the tf-idf matrix

'''dist is defined as 1 - the cosine similarity of each document.
Cosine similarity is measured against the tf-idf matrix and can be used to
generate a measure of similarity between each document and the other documents 
in the corpus. Subtracting it from 1 provides cosine distance which I will use 
for plotting on a euclidean (2-dimensional) plane.
Note that with dist it is possible to evaluate the similarity of any two or more texts
'''

dist = 1 - metrics.pairwise.cosine_similarity(tfidf_matrix)

##############################################################################
##############################################################################


''' HIERARCHICAL (DOCUMENT) CLUSTERING ANALYSIS
Ward clustering algorithm - agglomerative clustering method
step 1) used the precomputed cosine distance matrix (dist) to calclate a linkage_matrix
step 2) then plot as a dendrogram.
'''

def plot_dendrogram(dist):
    '''dist is a distance matrix created from the tf-idf'''
    linkage_matrix = hca.linkage(dist, method='ward') #using distance matrix fro tfidf
    
    fig, ax = plt.subplots(figsize=(6, 25)) # set size
    ax = hca.dendrogram(linkage_matrix, orientation='right', labels=sample.index.strftime('%Y-%m-%d'));
    
    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    
    plt.tight_layout() #show plot with tight layout

plot_dendrogram(dist)

##############################################################################
##############################################################################

num_clusters = input('How many clusters would you like?    ') # check dendrogram

##############################################################################
##############################################################################

#### K-means clustering ####
#num_clusters = len(sample['author'].unique()) # set number of clusters to number of unique authors

km = cluster.KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist() # list of clusters

# create doc attribute df including cluster assignment
revisions = { 'time': sample.index, 'author': sample['author'].tolist(), 'text': texts, 'cluster': clusters}
frame = pd.DataFrame(revisions, index = [clusters] , columns = ['time', 'author', 'cluster', 'text'])

print('Cluster value counts:')
frame['cluster'].value_counts()
print()

### identify n words most identified with each cluster, gives sense of cluster topic
def list_cluster_words():
    print("Top terms per cluster:")
    print()
    #sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
    
    for i in range(num_clusters):
        print("Cluster %d words:" % i, end='')
        
        for ind in order_centroids[i, :10]: # set num words to get
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        print() #add whitespace
        print() #add whitespace
        
        print("Cluster %d authors:" % i, end='')
        if type(frame.ix[i]['author']) == str:
            print(frame.ix[i]['author'])
        else:
            for author in set(frame.ix[i]['author']):
                print(' %s,' % author, end='')
        print() #add whitespace
        print() #add whitespace

list_cluster_words()

##############################################################################
##############################################################################
#### Multidimensional scaling ####
# convert the dist matrix into a 2-dimensional array using multidimensional scaling
# Another option would be to use principal component analysis.

manifold.MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

##############################################################################
##############################################################################

'''VISUALISING DOCUMENT CLUSTERS - K-MEANS'''

colors = sns.color_palette('viridis', num_clusters)                  
cluster_names = {}
# cannibalizing code above to get list of highest ranking words per cluster
for i in range(num_clusters):
    names = []
    for ind in order_centroids[i, :3]: 
        names.append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'))
    cluster_names[i] = names
    
node_attrs = pd.DataFrame(dict(x=xs, y=ys, label=clusters, date=sample.index.strftime('%m/%Y'), author=sample['author'])) #results of the MDS plus the cluster numbers and titles
                                                    #or title = sample['author']
groups = node_attrs.groupby('label') #group by cluster

## plot figure
def plot_clusters():
    fig, ax = plt.subplots(figsize=(9, 6)) # set size
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
                label=cluster_names[name], color=colors[name], 
                mec='none', alpha = 0.5)
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')
        
    ax.legend(numpoints=1, loc=0)  #show legend with only 1 point
    for i in range(len(node_attrs)): #add label in x,y position with the label as the author name
       ax.text(node_attrs.ix[i]['x'], node_attrs.ix[i]['y'], node_attrs.ix[i]['date'],
               size=10, rotation = 75, weight='semibold')  
            
plot_clusters()


##############################################################################
##############################################################################

''' TOPIC MODELLING - Latent Dirichlet Allocation
LDA is a probabilistic topic model that assumes documents are a mixture of
topics and that each word in the document is attributable to the document's topics
EXAMPLE REMOVED PROPER NOUNS, I HAVE NOT
'''

'''text is a list of sidebar revisions returns from prep_texts'''
#tokenized_text = [tokenize_and_stem(text) for text in texts] #tokenize
tokenized_text = [tokenize_only(text) for text in texts] #tokenize
stopless_texts = [[word for word in text if word not in stopwords] for text in tokenized_text] #remove stop words
dictionary = corpora.Dictionary(stopless_texts) #create a Gensim dictionary from the texts
dictionary.filter_extremes(no_below=20) #remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
corpus = [dictionary.doc2bow(text) for text in stopless_texts] #convert the dictionary to a bag of words corpus for reference

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


##############################################################################
##############################################################################
