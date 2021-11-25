# v0 -> tokenizes strings
# v1 -> adds return calculations
# v2 -> Saves bert outputs independently
# v3 -> One-hot encodes labels
# v4 -> Returns to single label

#%% Imports
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

#%% Function definitions
# Bert tokenization
def bert_tokenize_f(strings):
    token_ids = []
    token_types = []
    att_masks = []
    
    for string in tqdm(strings):
        #print(f'String = {string}')
        bert_out = bert_tokenizer(string,
                                  return_tensors = 'pt',
                                  padding = 'max_length',
                                  truncation = True,
                                  max_length = seq_len)
    
        token_ids.append(bert_out['input_ids'].squeeze(0).type(torch.LongTensor))
        token_types.append(bert_out['token_type_ids'].squeeze(0).type(torch.LongTensor))
        att_masks.append(bert_out['attention_mask'].squeeze(0).type(torch.LongTensor))
     
    return token_ids, token_types, att_masks

#%% Label generation
def generate_labels_f(returns):
    labels = []
    for ret in returns:
        ret = round(ret, 2)
        if pd.isnull(ret):
            label = 'NaN'
        elif ret <= 1:
            label = 0
        else:
            label = 1
        labels.append(label)

    return labels            

#%% Path definition
#input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/01_structured'
#output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary'

input_folder = '/data/rsg/nlp/sibanez/00_temp/01_fin_pred/00_data/01_structured'
output_folder = '/data/rsg/nlp/sibanez/00_temp/01_fin_pred/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary' 

input_path = os.path.join(input_folder, 'structured_2019_mapped_SP500_full.pkl')
output_pkl_path = os.path.join(output_folder, 'preproc_2019_mapped_SP500_full.pkl')

#%% Global initialization
seq_len = 256
max_n_pars_facts = 50
pad_int = 0
toy_data = False
len_toy_data = 100
model_name = 'ProsusAI/finbert'
pd.set_option('display.max_rows', 100)

#%% Data loading
data_df = pd.read_pickle(input_path)
if toy_data == True: data_df = data_df[0:len_toy_data]

#%% Tokenizer instantiation
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
    
#%% Tokenize headlines
headlines = data_df['headline'].astype(str)
headlines = headlines.replace('nan', '')
token_ids_headline, token_types_headline, att_masks_headline = \
    bert_tokenize_f(headlines)

#%% Compute returns
ret_p1 = data_df['close+1'] / data_df['close']
ret_p0 = data_df['close'] / data_df['close-1']
ret_m1 = data_df['close-1'] / data_df['close-2']
ret_m2 = data_df['close-2'] / data_df['close-3']
ret_m3 = data_df['close-3'] / data_df['close-4']
ret_m4 = data_df['close-4'] / data_df['close-5']

#%% Generate labels
labels_p1 = generate_labels_f(ret_p1)
labels_p0 = generate_labels_f(ret_p0)
labels_m1 = generate_labels_f(ret_m1)
labels_m2 = generate_labels_f(ret_m2)
labels_m3 = generate_labels_f(ret_m3)
labels_m4 = generate_labels_f(ret_m4)

#%% Build output dataframe
output_df = data_df[['amerTimestamp_x',
                     'assetName',
                     'headline']].copy()

output_df['token_ids_headline'] = token_ids_headline
output_df['token_types_headline'] = token_types_headline
output_df['att_masks_headline'] = att_masks_headline

output_df['labels_p1'] = labels_p1
output_df['labels_p0'] = labels_p0
output_df['labels_m1'] = labels_m1
output_df['labels_m2'] = labels_m2
output_df['labels_m3'] = labels_m3
output_df['labels_m4'] = labels_m4

#%% Check label value counts
print(pd.value_counts(output_df.labels_p1))
print(pd.value_counts(output_df.labels_p0))
print(pd.value_counts(output_df.labels_m1))
print(pd.value_counts(output_df.labels_m2))
print(pd.value_counts(output_df.labels_m3))
print(pd.value_counts(output_df.labels_m4))

#%% Save outputs
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

print(f'Saving datasets to {output_pkl_path}')
output_df.to_pickle(output_pkl_path)
print('Done')
