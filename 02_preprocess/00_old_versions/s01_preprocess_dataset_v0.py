# v0 -> tokenizes strings

#%% Imports
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer

#%% Path definition
input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/03_spy_project/00_data/01_preprocessed'
output_folder = input_folder

input_path = os.path.join(input_folder, 'preproc_2019_mapped_SP500.pkl')
output_pkl_path = os.path.join(output_folder, 'tokenized_2019_mapped_SP500.pkl')

#%% Global initialization
seq_len = 256
max_n_pars_facts = 50
pad_int = 0
toy_data = False
len_toy_data = 100
model_name = 'nlpaueb/legal-bert-small-uncased'

#%% Data loading
data_df = pd.read_pickle(input_path)
if toy_data == True: data_df = data_df[0:len_toy_data]

#%% Tokenizer instantiation
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
    
#%% Tokenize headlines
print(f'\n{datetime.now()} Tokenizing headline')
headlines = data_df['headline'].astype(str)
headlines = headlines.replace('nan', '')
tokens_bert = []

for headline in tqdm(headlines):
    print(f'Headline = {headline}')
    tokens_bert_aux = bert_tokenizer(headline,
                                     return_tensors = 'pt',
                                     padding = 'max_length',
                                     truncation = True,
                                     max_length = seq_len)
    tokens_bert.append(tokens_bert_aux)

print(f'{datetime.now()} Done')

#%% Build output dataframe
data_df['tokens'] = tokens_bert
data_df = data_df[['tokens',
                  'close']]

#%% Save outputs
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

print(f'Saving datasets to {output_folder}')
data_df.to_pickle(output_pkl_path)
print('Done')
