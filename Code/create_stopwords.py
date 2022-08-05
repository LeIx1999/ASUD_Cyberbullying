import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
# Combine german stopwords from nltk with additional stopwords
german_stopwords = []
with open('./Notebooks/data/stop_words_german.txt', encoding="utf-8") as f:
    for line in f:
        line = line.replace('\n', '')
        german_stopwords.append(line)

# add them up
german_stopwords = german_stopwords + stopwords.words("german")

# save stopwords
with open('./Notebooks/data/stopwords_all.txt', 'w', encoding="utf-8") as fp:
    for item in german_stopwords:
        fp.write("%s\n" % item)