from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import W2Vec_functions
import RF_functions
import pandas as pd

# read in data
data = pd.read_csv("/Users/jannis/ASUD_Cyberbullying/Notebooks/data/prepared_dataframe.csv")

# split data in training and test
data_train = data.sample(round(0.75 * len(data)), random_state = 1).reset_index()
data_test = data[~data.index.isin(data_train.index)].reset_index()

# spacy pipeline
tweet_words = W2Vec_functions.spacy_pipe(data_train["tweets_clean"], 3)

#create a word2vec model
word2vec = Word2Vec(tweet_words, min_count=3, sg=1, hs=0, negative=9,
                    ns_exponent=0.69, window=6, vector_size=60, epochs=80)

# train the model
word2vec.train(tweet_words, total_examples=word2vec.corpus_count, epochs=word2vec.epochs)

words_dict = W2Vec_functions.remove_nan_tweets(tweet_words, word2vec)

# subset data_pre and tweet_words, only tweets where at least one word is in the word2vec model
data_train = data_train.iloc[list(words_dict.keys()), :]
data_train = data_train.reset_index()
tweet_words = [tweet_words[i] for i in list(words_dict.keys())]

# get vector for each tweet
tweet_vectors = []
for tweet in tweet_words:
    vec = W2Vec_functions.get_com_vector(tweet, word2vec)
    tweet_vectors.append(vec)

# Modelling
# create tweet vectors of test tweets
new_tweet_v = []
words_new = []
for tweet in data_test.tweets_clean:
    new_tweet_v.append(W2Vec_functions.new_tweet_vector(tweet, word2vec)[0])
    words_new.append(W2Vec_functions.new_tweet_vector(tweet, word2vec)[1])

# Remove tweets where there are no words in the word2vec model
words_dict = W2Vec_functions.remove_nan_tweets(words_new, word2vec)

# subset data_test and tweet_words, only tweets where at least one word is in the word2vec model
data_test = data_test.iloc[list(words_dict.keys()), :]
data_test = data_test.reset_index(drop=True)
new_tweet_v = [new_tweet_v[i] for i in list(words_dict.keys())]

# Encode the classes
y_transformed, encoder = RF_functions.encode_classes(data_train["granulareKlassifikation"])

# Train a Random Forest
rfc = RandomForestClassifier()
rfc.fit(tweet_vectors, y_transformed)
rfc_predictions = rfc.predict(new_tweet_v)

# Decode classes
rfc_predictions = RF_functions.decode_classes(rfc_predictions, encoder)

# calculate accuracy
acc = RF_functions.calc_accuracy(rfc_predictions, data_test["granulareKlassifikation"])
print(acc)
# classification report
classification_report(data_test["granulareKlassifikation"], rfc_predictions)

# compare value counts
print(RF_functions.compare_value_counts(rfc_predictions, data_test["granulareKlassifikation"]))







