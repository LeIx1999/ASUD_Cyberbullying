import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import spacy
nlp = spacy.load('de_core_news_lg')

# function to create a spacy nlp pipeline and lemmatized words
def spacy_pipe(data: list, min_word_length=3) -> list:
    # create spacy docs with a nlp pipeline
    # https://spacy.io/usage/processing-pipelines
    docs = [nlp(tweet) for tweet in data]

    # extract the generated lemmas for words that are no stopwords and have a length of more than two
    # Lemmatization is the process of reducing inflected forms of a word while still ensuring that the reduced
    # form belongs to the language. This reduced form or root word is called a lemma.
    words = [
        [
            word.lemma_ for word in doc if ((not word.is_stop) and (len(word) >= min_word_length))
        ]
        for doc in docs]

    return words


# Remove tweets where there are no words in the word2vec model
def remove_nan_tweets(tweet_words: list, model) -> dict:
    tweet_words_dict = {}
    for tweet in tweet_words:
        for word in tweet:
            if word in model.wv.index_to_key:
                tweet_words_dict[tweet_words.index(tweet)] = True
    return tweet_words_dict

def get_sim(word_a: str, word_b: str, model) -> str:
    return f'{word_a} | {word_b}: {model.wv.similarity(word_a, word_b)}'

# test2 word similarities
def get_most_sim_words(word_list: list, model, top_n: int) -> str:
    return f'{word_list}: {model.wv.most_similar(positive=word_list, negative=[], topn = top_n)}'

# calculate tweet vectors
def get_com_vector(words : list, model) -> np.array:
    # list of words in the word2vec model
    words = [word for word in words if word in model.wv.index_to_key]
    # get the vectors
    vectors = np.array([model.wv.get_vector(word) for word in words])
    # return the sum of all vectors devided by the amount of words from words in the model
    vector = np.sum(vectors, axis=0)
    return vector / len(words)

# function to preprocess and transform new tweets
def new_tweet_vector(tweet : str, model):
    # call spacy pipeline function
    words_new = spacy_pipe([tweet], min_word_length = 3)[0]

    # calculate vector for new article
    new_tweet_v = get_com_vector(words_new, model)
    return new_tweet_v, words_new