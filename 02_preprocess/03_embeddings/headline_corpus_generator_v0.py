# v0

#%% Imports
import os
import nltk
import codecs
import fasttext
import unidecode
import pandas as pd
from datetime import datetime

#%% Path definitions
# Local
#input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/03_FULL-MA-final'
#output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/03_FT_LM'

# Server
input_folder = '/home/sibanez/Projects/00_MyInvestor/00_data/00_raw/03_FULL-MA-final'
output_folder = '/home/sibanez/Projects/00_MyInvestor/00_data/02_preprocessed/03_FT_LM'

#%% Global initialization
input_file_name = 'test.pkl'
output_corpus_file_name = 'corpus.txt'
output_LM_file_name = 'FT_LM.bin'

#%% Data loading
input_path = os.path.join(input_folder, input_file_name)
print(f'{datetime.now()} Loading data')
data_df = pd.read_pickle(input_path)
print(f'{datetime.now()} Done')

#%% Corpus generation
headlines = data_df.headline.to_list()
headlines = [x for sublist in headlines for x in sublist]
headlines = [x for x in headlines if x != '']
headlines = [unidecode.unidecode(x) for x in headlines]
headlines = [x for x in headlines if 'IMBALANCE' not in x]
headlines = ' '.join(headlines)
headlines = headlines.lower()

#%% Check number of tokens in corpus
tokens = nltk.word_tokenize(headlines)
unique_tokens = list(set(tokens))
print(f'Number of tokens = {len(tokens):,}')
print(f'Number of unique tokens = {len(unique_tokens):,}')

#%% Save corpus
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

output_corpus_path = os.path.join(output_folder, output_corpus_file_name)
with codecs.open(output_corpus_path, 'w', 'utf-8') as fw:
    fw.write(headlines)

#%% LM generation
model = fasttext.train_unsupervised(output_corpus_path)

#%% Save LM
output_LM_path = os.path.join(output_folder, output_LM_file_name)
model.save_model(output_LM_path)
