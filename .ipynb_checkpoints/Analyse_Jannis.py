import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Daten lesen
data = pd.read_csv("/Users/jannis/ASUD_Cyberbullying/datframe_subtask1_2.csv")

# Data Understanding
# print(data.values)
print(data.head())

# Summary Stats
print(data.describe())
bin_cat = pd.DataFrame(data["binaereKlassifikation"].value_counts())
gran_cat = pd.DataFrame(data["granulareKlassifikation"].value_counts())

# NAs
print(pd.isnull(data).sum())

# Anzahl Wörter pro Message
OTHER = data[data["binaereKlassifikation"] == "OTHER"]
OFFENSE = data[data["binaereKlassifikation"] == "OFFENSE"]
Word_count_other = [len(message.split()) for message in OTHER["tweet"]]
Word_count_offense = [len(message.split()) for message in OFFENSE["tweet"]]
Word_count = pd.DataFrame({'value': [round(st.mean(Word_count_other), 2),
                                        round(st.mean(Word_count_offense), 2)],
                           'Category': ["Other", "Offense"]})


# Bar chart
Word_count.plot.bar(x = 'Category', y = 'value', rot = 0, title = "Anzahl Wörter pro Tweet")
plt.show()

# Wordcloud ohne stopwords
german_stopwords = stopwords.words("german")
sw = set(STOPWORDS)
wc_OTHER = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords=german_stopwords).generate(str(OTHER["tweet"].values))
wc_OFFENSE = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords=german_stopwords).generate(str(OFFENSE["tweet"].values))

plt.imshow(wc_OTHER, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud OTHER")
# plt.savefig("Wordcloud_true.png")
plt.show()

plt.imshow(wc_OFFENSE, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud OFFENSE")
# plt.savefig("Wordcloud_spam.png")
plt.show()


# save plots
