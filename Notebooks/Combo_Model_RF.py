import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import spacy


nlp = spacy.load('de_core_news_lg')

# read in data
data = pd.read_csv("data/prepared_dataframe.csv")

# split data in training and test
data_train = data.sample(round(0.75 * len(data)), random_state = 1).reset_index()
data_test = data[~data.index.isin(data_train.index)].reset_index()

# train a word2vec model
# create spacy docs with a nlp pipeline
tweet_docs = [nlp(tweet) for tweet in data_train["tweets_clean"]]

# extract the generated lemmas for words that are no stopwords and have a length of more than two
# Lemmatization is the process of reducing inflected forms of a word while still ensuring that the reduced
# form belongs to the language. This reduced form or root word is called a lemma.
tweet_words = [
    [
    word.lemma_ for word in doc if ((not word.is_stop) and (len(word) >= 3))
    ]
    for doc in tweet_docs]

# create a word2vec model (gensim)
# https://developpaper.com/gensim-model-parameters-of-word2vec/
# Window: refers to the window size of training. 8 means that the first 8 words and the last 8 words are considered
word2vec = Word2Vec(tweet_words, min_count=3, sg=1, hs=0, negative=9,
                    ns_exponent=0.69, window=6, vector_size=60, epochs=80)

# train the model
word2vec.train(tweet_words, total_examples=word2vec.corpus_count, epochs=word2vec.epochs)


# Remove tweets where there are no words in the word2vec model
def remove_nan_tweets(tweets, model):
    tweet_words_dict = {}
    for tweet in tweets:
        sum_occurr = 0
        for word in tweet:
            if word in model:
                sum_occurr += 1
        if sum_occurr > 0:
            tweet_words_dict[tweets.index(tweet)] = True

    return tweet_words_dict


tweet_words_dict = remove_nan_tweets(tweet_words, word2vec.wv.index_to_key)

# subset data_pre and tweet_words, only tweets where at least one word is in the word2vec model
data_train = data_train.iloc[list(tweet_words_dict.keys()), :]
data_train = data_train.reset_index(drop=True)
tweet_words = [tweet_words[i] for i in list(tweet_words_dict.keys())]

# test word similarities
def test_word_sim(word_a, word_b):
    print(f'{word_a} | {word_b}: {word2vec.wv.similarity(word_a, word_b)}')

# test word similarities
def test_word_sim(word_a, word_b):
    print(f'{word_a} | {word_b}: {word2vec.wv.similarity(word_a, word_b)}')

# calculate tweet vectors
# calculate center of mass vector for list of words
def get_com_vector(words : list) -> np.array:
    # list of words in the word2vec model
    words = [word for word in words if word in word2vec.wv.index_to_key]
    # get the vectors
    vectors = np.array([word2vec.wv.get_vector(word) for word in words])
    # return the sum of all vectors devided by the amount of words from words in the model
    vector = np.sum(vectors, axis=0)
    return vector / len(words)

# get vector for each tweet
tweet_vectors = []
for tweet in tweet_words:
    vec = get_com_vector(tweet)
    tweet_vectors.append(vec)

# function to preprocess and transform new tweets
def new_tweet_vector(tweet : str):
    doc_new = nlp(tweet)

    words_new = [
                    word.lemma_ for word in doc_new
                    if (not word.is_stop) and (len(word)>2)
                 ]

    # calculate vector for new article
    new_tweet_v = get_com_vector(words_new)

    return new_tweet_v, words_new

# Modelling part 1
# create tweet vectors of test tweets
new_tweet_v = []
words_new = []
for tweet in data_test.tweets_clean:
    new_tweet_v.append(new_tweet_vector(tweet)[0])
    words_new.append(new_tweet_vector(tweet)[1])

# Remove tweets where there are no words in the word2vec model
tweet_words_dict = remove_nan_tweets(words_new, word2vec.wv.index_to_key)

# subset data_test and tweet_words, only tweets where at least one word is in the word2vec model
data_test = data_test.iloc[list(tweet_words_dict.keys()), :]
data_test = data_test.reset_index(drop=True)
new_tweet_v = [new_tweet_v[i] for i in list(tweet_words_dict.keys())]

# Encode the classes
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_transformed = encoder.fit_transform(data_train["binaereKlassifikation"])
encoder.classes_

# Train a Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(tweet_vectors, y_transformed)
rfc_predictions = rfc.predict(new_tweet_v)

# transform predictions to classes
encoder_dict = dict(enumerate(encoder.classes_.flatten(), 0))
rfc_predictions = [encoder_dict[x] for x in rfc_predictions]

# evaluation part 1
# calculate accuracy
def calc_accuracy(preds, act_values):
    acc = 0
    for i in range(len(preds)):
        if preds[i] == act_values[i]:
            acc += 1
    accuracy = acc / len(preds)
    return f'Accuracy: {str(accuracy)}'

calc_accuracy(rfc_predictions, data_test["binaereKlassifikation"])

# classification report
from sklearn.metrics import classification_report
print(classification_report(data_test["binaereKlassifikation"], rfc_predictions))

# Modelling Part 2
# Only train on offensive tweets
data_train_gran = data_train[data_train["granulareKlassifikation"] != "OTHER"]
tweet_vectors_gran = [tweet_vectors[i] for i in data_train_gran.index.values.tolist()]
data_train_gran = data_train_gran.reset_index(drop=True)

# create tweet vectors of test tweets
# only offencsive tweets
data_test_gran = data_test[data_test["granulareKlassifikation"] != "OTHER"]
data_test_gran = data_test_gran.reset_index(drop=True)

new_tweet_v = []
words_new = []
for tweet in data_test_gran.tweets_clean:
    new_tweet_v.append(new_tweet_vector(tweet)[0])
    words_new.append(new_tweet_vector(tweet)[1])

# Remove tweets where there are no words in the word2vec model
tweet_words_dict = remove_nan_tweets(words_new, word2vec.wv.index_to_key)

# subset data_test and tweet_words, only tweets where at least one word is in the word2vec model
data_test_gran = data_test_gran.iloc[list(tweet_words_dict.keys()), :]
data_test_gran = data_test_gran.reset_index(drop=True)
new_tweet_v_gran = [new_tweet_v[i] for i in list(tweet_words_dict.keys())]

# Encode the classes
from sklearn.preprocessing import LabelEncoder
encoder_gran = LabelEncoder()
y_transformed_gran = encoder_gran.fit_transform(data_train_gran["granulareKlassifikation"])
encoder_gran.classes_

# Train a Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc_gran = RandomForestClassifier()
rfc_gran.fit(tweet_vectors_gran, y_transformed_gran)
rfc_predictions_gran = rfc_gran.predict(new_tweet_v_gran)

# transform predictions to classes
encoder_dict_gran = dict(enumerate(encoder_gran.classes_.flatten(), 0))
rfc_predictions_gran = [encoder_dict_gran[x] for x in rfc_predictions_gran]

# Evaluation Part 2
# calculate accuracy
calc_accuracy(rfc_predictions_gran, data_test_gran["granulareKlassifikation"])

# classification report
from sklearn.metrics import classification_report
print(classification_report(data_test_gran["granulareKlassifikation"], rfc_predictions_gran))

# Combine both models
def make_predictions(tweet):
    # create tweet vectors of test tweets
    new_tweet_v = new_tweet_vector(tweet)[0]

    # make binaere predictions
    rfc_predictions = rfc.predict([new_tweet_v])

    encoder_dict = dict(enumerate(encoder.classes_.flatten(), 0))
    rfc_predictions = [encoder_dict[x] for x in rfc_predictions]

    if rfc_predictions[0] == "OTHER":
        return rfc_predictions[0]
    else:
        # make granulare predictions
        rfc_predictions_gran = rfc_gran.predict([new_tweet_v])

        encoder_dict_gran = dict(enumerate(encoder_gran.classes_.flatten(), 0))
        rfc_predictions_gran = [encoder_dict_gran[x] for x in rfc_predictions_gran]
        return rfc_predictions_gran[0]

# Evaluation of combo model
combo_preds = [make_predictions(x) for x in data_test.tweets_clean]

calc_accuracy(combo_preds, data_test["granulareKlassifikation"])
