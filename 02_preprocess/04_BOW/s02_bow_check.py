# v0 -> Double checks the BOW encoding

#%% Imports
import os
import nltk
import pickle
import unidecode
import pandas as pd

#%% Path definitions
# Local
#input_folder = ''

# Server
input_folder = '/home/sibanez/Projects/00_MyInvestor/00_data/02_preprocessed/04_BOW/00_binary/01_ALL'

#%% Global initialization
entry = 6577
data_filename = 'model_test.pkl'
bow_filename = 'bow_vocab.pkl'

#%% Data loading
data_path = os.path.join(input_folder, data_filename)
bow_vocab_path = os.path.join(input_folder, bow_filename)

data_df = pd.read_pickle(data_path)
with open(bow_vocab_path, 'rb') as fr:
	bow_vocab = pickle.load(fr)

#%% Extract headlines
headlines = data_df.loc[entry]['headline']
headlines = [unidecode.unidecode(x) for x in headlines]
headlines = (' ').join(headlines)
headlines = headlines.lower()
tokens = nltk.word_tokenize(headlines)
print(tokens)

bow = data_df.loc[entry]['bow']
tokens_bow = []
for idx, x in enumerate(bow):
	if x == 1:
		tokens_bow.append(bow_vocab[idx])

print(tokens_bow)




