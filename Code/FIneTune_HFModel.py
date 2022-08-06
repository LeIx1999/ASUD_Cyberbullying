import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from transformers import DefaultDataCollator
from datasets import Dataset

# https://huggingface.co/docs/transformers/training
# read in data
data = pd.read_csv("/Users/jannis/ASUD_Cyberbullying/Notebooks/data/prepared_dataframe.csv")

enc_dict = {"OTHER": 0,
            "PROFANITY": 1,
            "ABUSE": 2,
            "INSULT": 3}

data["granulareKlassifikation"] = [enc_dict[x] for x in data["granulareKlassifikation"]]

data = data.rename(columns={"tweets_clean": "text",
                            "granulareKlassifikation": "label"})

# split data in training and test
data_train = data.sample(round(0.75 * len(data)), random_state = 1).reset_index()
data_test = data[~data.index.isin(data_train.index)].reset_index()

# load pretrained transformer model
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# pandas dfs to Datasets
data_train = Dataset.from_pandas(data_train[["label", "text"]])
data_test= Dataset.from_pandas(data_test[["label", "text"]])


# use tokenizer from pretrained model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_tokenized = data_train.map(tokenize_function, batched=True)
test_tokenized = data_test.map(tokenize_function, batched=True)

# train the model
model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", num_labels=4)

training_args = TrainingArguments(output_dir="test_trainer")

# load accuracy metric
metric = load_metric("accuracy")

# create function to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# monitor evaluation metrics
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
)

trainer.train()

# make predictions
preds = trainer.predict(test_tokenized)

# Convert to TensorFlow format
data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_dataset = train_tokenized.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

tf_validation_dataset = test_tokenized.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)

