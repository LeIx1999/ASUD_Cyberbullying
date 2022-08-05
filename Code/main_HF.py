from sentence_transformers import SentenceTransformer
import pandas as pd
import RF_functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump

# read in data
data = pd.read_csv("/Users/jannis/ASUD_Cyberbullying/Notebooks/data/prepared_dataframe.csv")

# split data in training and test
data_train = data.sample(round(0.75 * len(data)), random_state = 1).reset_index()
data_test = data[~data.index.isin(data_train.index)].reset_index()

# load pretrained transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# create tweet embeddings
tweet_vectors = model.encode(data_train["tweets_clean"])
new_tweet_v = model.encode(data_test["tweets_clean"])

# Encode the classes
y_transformed, encoder = RF_functions.encode_classes(data_train["granulareKlassifikation"])

# Train a Random Forest
rfc = RandomForestClassifier(random_state=42)
# hyperparameter = {"max_depth": [4, 6, 8, 10, 12],
#                  "min_samples_split": [2, 3, 4, 5, 6]}
# ,
#                   "max_samples": [0.6, 0.7, 0.8, 0.9, 1]

# rfc_grid = GridSearchCV(rfc, hyperparameter, cv=5)
rfc.fit(tweet_vectors, y_transformed)

# make predictions
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

# save the model as joblib, replacement of pickle to save to disk
# dump(rfc, "model.joblib")


