# v0 Takes a single dataset and generates train and dev bow sets

#%% Imports
import os
import nltk
import string
import unidecode
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

#%% Path definitions

# Laptop
#input_path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/04_FULL-MA-final_fixed/2018_headlines.pkl'
#output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/03_runs/03_BOW/00_TEST'

# Server
input_path = '/home/sibanez/Projects/00_MyInvestor/00_data/00_raw/04_FULL-MA-final-fixed/2018_headlines.pkl'
output_folder = '/home/sibanez/Projects/00_MyInvestor/00_data/02_preprocessed/04_BOW/00_binary/01_2018_no_imbalance'

output_path_train = os.path.join(output_folder, 'model_train.pkl')
output_path_dev = os.path.join(output_folder, 'model_dev.pkl')

#%% Global initialization
shuffle = False
bow_size = 2000
stopwords_en = stopwords.words('english')

#%% Function definitions
def generate_bow_f(headlines, bow_vocab, bow_size):
    
    text = (' ').join(headlines)
    text = unidecode.unidecode(text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [x for x in tokens if x not in stopwords_en]
    tokens = [x for x in tokens if x not in string.punctuation]
    
    bow = [0] * bow_size
    for token in tokens:
        if token in bow_vocab:
            bow[bow_vocab.index(token)] += 1
    
    return bow
            
#%% Dataset reading
print(f'{datetime.now()}: Reading data')
data_df = pd.read_pickle(input_path)
print(f'{datetime.now()}: Done')

#%% Check dataset size and years present
print(f'Shape dataset = {data_df.shape}\n')
years = [x.year for x in data_df.date]
print(f'Value counts years:\n{pd.value_counts(years)}')

#%% Split in train and dev sets
train_df, dev_df = train_test_split(data_df, test_size=0.2, shuffle = shuffle)

#%% Check dataset size
print(f'Shape train set = {train_df.shape}')
print(f'Shape dev set = {dev_df.shape}')

print(f'% train set = {len(train_df) / len(data_df):.2f}')
print(f'% dev set = {len(dev_df) / len(data_df):.2f}')

#%% Generate corpus
headlines = train_df.headline.to_list()
headlines = [x for sublist in headlines for x in sublist]
headlines = [x for x in headlines if x != '']
headlines = [unidecode.unidecode(x) for x in headlines]
headlines = ' '.join(headlines)
headlines = headlines.lower()

#%% Generate vocabulary
tokens = nltk.word_tokenize(headlines)
tokens = [x for x in tokens if x not in stopwords_en]
tokens = [x for x in tokens if x not in string.punctuation]
token_count = pd.value_counts(tokens)
print(f'\nToken count[0:20]:\n{token_count[0:20]}')

#%% Generate BOW
vocab = list(set(tokens))
bow_vocab = list(token_count.keys())[:bow_size]

#%% Slice datasets
selected_features = ['date',
                     'label1',
                     'label2',
                     'label3',
                     'label4',
                     'label5',
                     'headline']

train_df = train_df[selected_features]
dev_df = dev_df[selected_features]
print(f'Shape train set = {train_df.shape}')
print(f'Shape dev set = {dev_df.shape}')

#%% Generate BOWs
bow_aux = []
for headlines in tqdm(train_df.headline, desc='Generating BOW train'):    
    bow_aux.append(generate_bow_f(headlines, bow_vocab, bow_size))
train_df['bow'] = bow_aux

#%%
bow_aux = []
for headlines in tqdm(dev_df.headline, desc='Generating BOW dev'):    
    bow_aux.append(generate_bow_f(headlines, bow_vocab, bow_size))
dev_df['bow'] = bow_aux

#%% Save output datasets
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

print(f'{datetime.now()} Saving datasets')
train_df.to_pickle(output_path_train)
dev_df.to_pickle(output_path_dev)
print(f'{datetime.now()} Done')
