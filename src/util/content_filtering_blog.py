import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson.json_util import dumps
from datetime import datetime

import re, string, unicodedata, contractions, inflect

from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer, SnowballStemmer, PorterStemmer
from nltk.corpus import stopwords
from textblob import TextBlob, Word
from stemming.porter2 import stem

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import pearsonr

from subprocess import check_output
import bson
stop = stopwords.words('english')

class blog_content_filtering():
    def __init__(self):
        print()
    
    
    
    def content_blog(self, blogs):
        
        data_blog = blogs.fillna("")[["content", "subtitle", "tags", "title"]]
        df_tags = pd.DataFrame(data_blog["content"] + " " +data_blog["subtitle"] + " " +data_blog["tags"] + " " +data_blog["title"])

        # df_tags = pd.DataFrame(blogs.fillna("").apply(lambda x: x[3]+x[4]+x[5]+x[6], axis=1))
        df_tags.columns = ["content"]
        df_tags['raw_data'] = df_tags["content"]
        
        return df_tags
    
    
    ## removal noise and contractions to fix 
    def replace_noise_contraction(text):
        return contractions.fix(re.sub('\[[^]]*\]', '', BeautifulSoup(text, "html.parser").get_text()))


    ## word tokenzation
words = nltk.word_tokenize(sample)
# print(words)


## normalization
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

words = normalize(words)
# print(words)


## stemming and lemmantize
def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas

stems, lemmas = stem_and_lemmatize(words)
print('Stemmed:\n', stems)
print('\nLemmatized:\n', lemmas)
    
   

    def feature_extraction(self, df_tags):
        # word count
        df_tags['word_count'] = df_tags['content'].apply(lambda x: len(str(x).split(" ")))
        # char count
        df_tags['char_count'] = df_tags['content'].str.len()

        df_tags['avg_word'] = df_tags["content"].apply(avg_word)

        # stop words count
        df_tags["stopwords"] = df_tags["content"].apply(lambda x: len([y for y in x.split() if y in stop]))

        #hash tags
        df_tags["hashtags"] = df_tags["content"].apply(lambda x: len([y for y in x.split() if y.startswith('#')]))

        #numerics
        df_tags["numerics"] = df_tags["content"].apply(lambda x: len([y for y in x.split() if y.isdigit()]))

        #uppercase
        df_tags["uppercase"] = df_tags["content"].apply(lambda x: len([y for y in x.split() if y.isupper()]))
    
    
    
    ## 
    def 
    
    def preprocessing(self, df_tags):
        
        #transform content to lower case
        df_tags["content"] = df_tags["content"].apply(lambda x: " ".join(y.lower() for y in x.split()))

        # removing punctuation
        df_tags["content"] = df_tags["content"].str.replace('[^\w\s]', '')

        # removal of stopwords
        df_tags["content"] = df_tags["content"].apply(lambda x: " ".join(y for y in x.split() if y not in stop))

        #common word removal
        # freq = list(pd.Series(' '.join(df_tags["content"]).split()).value_counts()[:10].index)
        # df_tags["content"] = df_tags["content"].apply(lambda x: " ".join(y for y in x.split() if y not in freq))

        #Rare words removal
        # freq = list(pd.Series(" ".join(df_tags["content"]).split()).value_counts()[-10:].index)
        # df_tags['content'] = df_tags["content"].apply(lambda x: " ".join( y for y in x.split() if y not in freq))

        # spelling correction
        df_tags["content"] = df_tags["content"].apply(lambda x: str(TextBlob(x).correct()))

        # Tokenization
        # TextBlob(df_tags["content"][0]).words

        #stemming
        # df_tags["content"] = df_tags["content"].apply(lambda x: " ".join([stem(y) for y in x.split()]))

        #Lemmatization
        df_tags["content"] = df_tags["content"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    
    
    def text_processing_tfidf(self, df_tags):
        
        ## Adv Text Processing

        #N-gram N=2
        # ngramVec = df_tags["content"].apply(lambda x: TextBlob(x).ngrams(2))

        # Term Frequency
        # tf1 = df_tags["content"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        # tf1.columns = ["words", "tf"]

        # # Inverse document frequency
        # for i, word in enumerate(tf1["words"]):
        #     tf1.loc[i, 'idf'] = np.log(df_tags.shape[0]/(len(df_tags[df_tags["content"].str.contains(word)])))


        # Term frequency and inverse document frequency
        # tf1["tfidf"] = tf1["tf"]*tf1["idf"]


        # Term frequency and inverse document frequency with sklearn lib
        tfidf = TfidfVectorizer(max_features=1450, lowercase=True, analyzer="word", stop_words="english", ngram_range=(1,1))
        tfidf_vect = tfidf.fit_transform(df_tags["content"])

        df_tfidf = pd.DataFrame(tfidf_vect.toarray())
        df_tfidf.columns = tfidf.get_feature_names()

        # merge columns with tfidf
        df_tags_process = df_tags.merge(df_tfidf, left_index=True, right_index=True)

        #Bag of words
        # bow = CountVectorizer(max_features=300, lowercase=True, analyzer="word", ngram_range=(1,1))
        # train_bow = bow.fit_transform(df_tags["content"])
        # bow.get_feature_names()

        #Sentiment analysis
        df_tags_process["sentiment"] = df_tags["raw_data"].apply(lambda x: TextBlob(x).sentiment[0])
        return df_tags_process.drop(["content", "raw_data"], axis=1)


















































    # average words
    def avg_word(sentence):
        words = sentence.split()
        return (sum(len(word) for word in words)/len(words))