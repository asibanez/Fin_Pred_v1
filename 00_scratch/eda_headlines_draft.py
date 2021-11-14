import nltk
import pandas as pd
import matplotlib.pyplot as plt

headlines = data.headline.to_list()
headline_types = [type(x) for x in headlines]
pd.value_counts(headline_types)

tokens = [nltk.word_tokenize(x) if type(x) == str else [] for x in headlines]
lens = [len(x) for x in tokens]

plt.hist(lens)
max(lens)
