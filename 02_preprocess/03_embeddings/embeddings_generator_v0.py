# v0

#%% Imports
import os
import nltk
import pickle
import fasttext
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split

#%% Path definition
# Local
#input_path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/04_FULL-MA-final_fixed/2018_headlines.pkl'
#model_path = ''
#output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/03_FastText_LM/01_preprocessed'

# Server
input_path = '/home/sibanez/Projects/00_MyInvestor/00_data/00_raw/04_FULL-MA-final-fixed/2018_headlines.pkl'
model_path = '/home/sibanez/Projects/00_MyInvestor/00_data/02_preprocessed/03_FastText_LM/00_models/cc.en.300.bin'
output_folder = '/home/sibanez/Projects/00_MyInvestor/00_data/02_preprocessed/03_FastText_LM/01_preprocessed/2018'

output_path_id_2_vec_dict = os.path.join(output_folder, 'word_to_vec.pkl')
output_path_train = os.path.join(output_folder, 'model_train.pkl')
output_path_dev = os.path.join(output_folder, 'model_dev.pkl')

#%% Global initialization
seq_len = 64
shuffle = False

#%% Load model and read input data
print(f'{datetime.now()} Loading model and dataset')
model = fasttext.load_model(model_path)
data_df = pd.read_pickle(input_path)
print(f'{datetime.now()} Done')

#%% Generate id to vector dictionary
model_dim = model.get_dimension()
unk_token = '<unk>'
pad_token = '<pad>'
unk_vector = np.zeros(model_dim)
pad_vector = np.ones(model_dim)

id_2_vec = {}
id_2_vec[0] = unk_vector
id_2_vec[1] = pad_vector

word_list = model.get_words()

for idx, word in enumerate(tqdm(word_list)):
    word_vector = model.get_word_vector(word)
    id_2_vec[idx + 2] = word_vector

#%% Generate new dataset with token ids
data_df = data_df[['dscd',
                   'date',
                   'label1',
                   'label2',
                   'label3',
                   'label4',
                   'label5',
                   'headline',
                   'headline_1',
                   'headline_2',
                   'headline_3']]

aux_1 = []
for entry in tqdm(data_df.headline):
    aux_2 = []
    for headline in entry:
        headline = headline.lower()
        tokens = nltk.word_tokenize(headline)
        tokens = tokens[0:seq_len]
        tokens += [pad_token] * (seq_len - len(tokens))
        ids = [1 if x == pad_token else model.get_word_id(x) for x in tokens]
        ids = [x if x != -1 else 0 for x in ids]
        assert(len(ids) == seq_len)
        aux_2.append(ids)       
    aux_1.append(aux_2)
    
data_df['ids'] = aux_1

#%% Check dataset size and years present
print(f'Shape dataset = {data_df.shape}\n')
years = [x.year for x in data_df.date]
print(pd.value_counts(years))

#%% Dataset split
train_df, dev_df = train_test_split(data_df, test_size=0.2, shuffle = shuffle)

#%% Check dataset size
print(f'Shape train set = {train_df.shape}')
print(f'Shape dev set = {dev_df.shape}')

print(f'% train set = {len(train_df) / len(data_df):.2f}')
print(f'% dev set = {len(dev_df) / len(data_df):.2f}')

#%% Save id_to_vector dictionary
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

with open(output_path_id_2_vec_dict, 'wb') as fw:
    pickle.dump(id_2_vec, fw, protocol = pickle.HIGHEST_PROTOCOL)

#%% Save output datasets
print(f'{datetime.now()} Saving datasets')
train_df.to_pickle(output_path_train)
dev_df.to_pickle(output_path_dev)
print(f'{datetime.now()} Done')


