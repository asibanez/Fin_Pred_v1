# Generates datasets for attention models:
#   output_df_1 ---> 1D attention over #news
#   output_df_2 ---> 2D attention over #news and #days

#%% Imports
import arrow
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

#%% Path definition
path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/00_toy_dataset/preproc_2019_mapped_SP500_full.pkl'

#%% Global initialization
days_considered = 3
news_considered = 3
model_name = 'ProsusAI/finbert'
seq_len = 256

#%% Load datasets
data = pd.read_pickle(path)

#%% Process dates
dates = data.amerTimestamp_x
dates_new = [arrow.get(x).date() for x in dates]
data['date'] = dates_new

#%% Sort dataset
data_sorted = data.sort_values(by = ['assetName', 'date'])
asset_list = sorted(list(set(data_sorted.assetName)))
date_list = sorted(list(set(data_sorted.date)))

#%% EDA
sliced = data_sorted[['date', 'assetName', 'headline']]
group = sliced.groupby(['date','assetName']).count()
print(group)
_ = plt.hist(list(group.headline), bins = 50)

#%% Compute empty headline
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
empty = bert_tokenizer('',
                       return_tensors = 'pt',
                       padding = 'max_length',
                       truncation = True,
                       max_length = seq_len)
    
empty_token_ids = (empty['input_ids'].squeeze(0).type(torch.LongTensor))
empty_token_types = (empty['token_type_ids'].squeeze(0).type(torch.LongTensor))
empty_att_masks = (empty['attention_mask'].squeeze(0).type(torch.LongTensor))

#%% Generate output_df_1 (1D attention over n news in 1 day)
output_df_1 = pd.DataFrame(columns = ['date',
                                    'asset',
                                    'headlines',
                                    'token_ids',
                                    'token_types',
                                    'att_masks',
                                    'labels'])

#for asset in tqdm(asset_list[0:3]): ==== TOY DATASET ====
for asset in tqdm(asset_list):    
    data_aux_1 = data_sorted[data_sorted.assetName == asset]
    for date in date_list:
        headlines = []
        token_ids = []
        token_types = []
        att_masks = []
        data_aux_2 = data_aux_1[data_aux_1.date == date]
        news_available = data_aux_2.shape[0]
        if news_available > 0:
            labels = list(data_aux_2.iloc[0][['labels_p1', 'labels_p0',
                                              'labels_m1', 'labels_m2',
                                              'labels_m3', 'labels_m4']])
        else:
            labels = ['','', '', '', '']
            
        for news_idx in range(news_considered):
            if news_idx < news_available:
                headlines.append(data_aux_2.iloc[news_idx]['headline'])
                token_ids.append(data_aux_2.iloc[news_idx]['token_ids_headline'])
                token_types.append(data_aux_2.iloc[news_idx]['token_types_headline'])
                att_masks.append(data_aux_2.iloc[news_idx]['token_types_headline'])
            else:
                headlines.append('')
                token_ids.append(empty_token_ids)
                token_types.append(empty_token_types)
                att_masks.append(empty_att_masks)
    
        row = pd.Series([date, asset, headlines, token_ids,
                         token_types, att_masks, labels],
                        index = output_df_1.columns)
        output_df_1 = output_df_1.append(row, ignore_index = True)
    
#%% Generate output_df_2 (2D attention over n news and m days)
output_df_2 = pd.DataFrame(columns = ['date',
                                    'asset',
                                    'headlines',
                                    'token_ids',
                                    'token_types',
                                    'att_masks',
                                    'labels'])

for asset in tqdm(asset_list):
    data_aux_1 = output_df_1[output_df_1.asset == asset]
    for idx in range(days_considered-1, len(data_aux_1)):
        date = data_aux_1.iloc[idx]['date']
        labels = data_aux_1.iloc[idx]['labels']
        headlines = list(data_aux_1.iloc[idx-(days_considered-1):idx+1]['headlines'])
        token_ids = list(data_aux_1.iloc[idx-(days_considered-1):idx+1]['token_ids'])
        token_types = list(data_aux_1.iloc[idx-(days_considered-1):idx+1]['token_types'])
        att_masks = list(data_aux_1.iloc[idx-(days_considered-1):idx+1]['att_masks'])
        row = pd.Series([date, asset, headlines, token_ids,
                         token_types, att_masks, labels],
                        index = output_df_2.columns)
        output_df_2 = output_df_2.append(row, ignore_index = True)
