# v0 -> Takes train, dev and test sets
#       Generates BOW vocabulary from train set
#       Generates train, dev and test BOW encodings

#%% Imports
import os
import nltk
import pickle
import string
import unidecode
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from nltk.corpus import stopwords

#%% Path definitions

# Laptop
#input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/04_FULL-MA-final_fixed/'
#output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/03_runs/03_BOW/00_TEST'

# Server
input_folder = '/home/sibanez/Projects/00_MyInvestor/00_data/00_raw/00_FULL-MA-final-fixed/01_train_dev_test'
output_folder = '/home/sibanez/Projects/00_MyInvestor/00_data/02_preprocessed/04_BOW/00_binary/01_ALL'

input_path_train = os.path.join(input_folder, 'train.pkl')
input_path_dev = os.path.join(input_folder, 'dev.pkl')
input_path_test = os.path.join(input_folder, 'test.pkl')

output_path_train = os.path.join(output_folder, 'model_train.pkl')
output_path_dev = os.path.join(output_folder, 'model_dev.pkl')
output_path_test = os.path.join(output_folder, 'model_test.pkl')
output_path_bow_vocab = os.path.join(output_folder, 'bow_vocab.pkl')

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
print(f'{datetime.now()}: Reading train set')
train_df = pd.read_pickle(input_path_train)
print(f'{datetime.now()}: Reading dev set')
dev_df = pd.read_pickle(input_path_dev)
print(f'{datetime.now()}: Reading test set')
test_df = pd.read_pickle(input_path_test)
print(f'{datetime.now()}: Done')

#%% Check dataset size and years present
print(f'\nShape train_set = {train_df.shape}\n')
print(f'Shape dev_set = {dev_df.shape}\n')
print(f'Shape test_set = {test_df.shape}\n')

years_train = [x.year for x in train_df.date]
years_dev = [x.year for x in dev_df.date]
years_test = [x.year for x in test_df.date]
print(f'\nValue counts years train:\n{pd.value_counts(years_train).sort_index()}')
print(f'Value counts years dev:\n{pd.value_counts(years_dev).sort_index()}')
print(f'Value counts years test:\n{pd.value_counts(years_test).sort_index()}')

size_full_set = len(train_df) + len(dev_df) + len(test_df)

print(f'\n% train set = {len(train_df) / size_full_set:.2f}')
print(f'\n% dev set = {len(dev_df) / size_full_set:.2f}')
print(f'\n% test set = {len(test_df) / size_full_set:.2f}')

#%% Generate corpus from train set
print(f'{datetime.now()}: Generating corpus from train set')
headlines = train_df.headline.to_list()
headlines = [x for sublist in headlines for x in sublist]
headlines = [x for x in headlines if x != '']
headlines = [unidecode.unidecode(x) for x in headlines]
headlines = ' '.join(headlines)
headlines = headlines.lower()
print(f'{datetime.now()}: Done')

#%% Generate vocabulary from train set
print(f'{datetime.now()}: Generating vocabulary from train set')
tokens = nltk.word_tokenize(headlines)
tokens = [x for x in tokens if x not in stopwords_en]
tokens = [x for x in tokens if x not in string.punctuation]
token_count = pd.value_counts(tokens)
print(f'\nToken count[0:20]:\n{token_count[0:20]}')

#%% Generate BOW vocab from train set
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
test_df = test_df[selected_features]
print(f'Shape train set = {train_df.shape}')
print(f'Shape dev set = {dev_df.shape}')
print(f'Shape test set = {test_df.shape}')

#%% Generate BOWs for train set
bow_aux = []
for headlines in tqdm(train_df.headline, desc='Generating BOW train'):    
    bow_aux.append(generate_bow_f(headlines, bow_vocab, bow_size))
train_df['bow'] = bow_aux

#%% Generate BOWs for dev set
bow_aux = []
for headlines in tqdm(dev_df.headline, desc='Generating BOW dev'):    
    bow_aux.append(generate_bow_f(headlines, bow_vocab, bow_size))
dev_df['bow'] = bow_aux

#%% Generate BOWs for test set
bow_aux = []
for headlines in tqdm(test_df.headline, desc='Generating BOW test'):    
    bow_aux.append(generate_bow_f(headlines, bow_vocab, bow_size))
test_df['bow'] = bow_aux

#%% Save output datasets
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

with open(output_path_bow_vocab, 'wb') as fw:
	pickle.dump(bow_vocab, fw)

print(f'{datetime.now()} Saving train set')
train_df.to_pickle(output_path_train)
print(f'{datetime.now()} Saving dev set')
dev_df.to_pickle(output_path_dev)
print(f'{datetime.now()} Saving test set')
test_df.to_pickle(output_path_test)
print(f'{datetime.now()} Done')
