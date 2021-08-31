# v0 -> Tokenize preprocessed datasets

#%% Imports
import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#%% Global initialization
pd.set_option('display.max_rows', 100)

#%% Path definitions
input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/03_spy_project/00_data/01_preprocessed'
output_folder = input_folder

input_path = os.path.join(input_folder, 'preproc_2019_mapped_SP500.pkl')
output_pkl_path = os.path.join(output_folder, 'tokenized_2019_mapped_SP500.pkl')
output_csv_path = os.path.join(output_folder, 'tokenized_2019_mapped_SP500.csv')

#%% Load dataset
data_df = pd.read_pickle(input_path)

#%% Check dataset size
print(f'Size dataset = {data_df.shape}')

#%% Tokenize headline
print(f'\n{datetime.now()} Tokenizing headline')
headline = data_df['headline'].astype(str)
headline = headline.replace('nan', '')
headline = [nltk.word_tokenize(x.lower()) for x in headline]
print(f'{datetime.now()} Done')
data_df['headline'] = headline

#%% Tokenize body
print(f'\n{datetime.now()} Tokenizing body')
body = data_df['body'].astype(str)
body = body.replace('nan', '')
body = [nltk.word_tokenize(x.lower()) for x in body]
print(f'{datetime.now()} Done')
data_df['body'] = body

#%% Check dataset size
print(f'Size dataset = {data_df.shape}')

#%% Save output dataset
data_df.to_pickle(output_pkl_path)
data_df.to_csv(output_csv_path)

#%% Scratch
bodies = [x for x in data_df.body if x != '']

#%%
data_df_final = data_df[['id',
                         'amerTimestamp_x',
                         'assetName',
                         'assetClass',
                         'sentimentClass',
                         'sentimentNegative',
                         'sentimentNeutral',
                         'sentimentPositive',
                         'data.messageType',
                         'data.mimeType',
                         'headline',
                         'body']]

#%%
data_df_1 = data_df_final[data_df_final['data.messageType'] == 1]
data_df_2 = data_df_final[data_df_final['data.messageType'] == 2]
data_df_5 = data_df_final[data_df_final['data.messageType'] == 5]
data_df_6 = data_df_final[data_df_final['data.messageType'] == 6]
data_df_7 = data_df_final[data_df_final['data.messageType'] == 7]

#%%
lens_1 = [len(x) for x in data_df_1.body]
lens_2 = [len(x) for x in data_df_2.body]
lens_5 = [len(x) for x in data_df_5.body]
lens_6 = [len(x) for x in data_df_6.body]
lens_7 = [len(x) for x in data_df_7.body]

#%%
plt.hist(lens_1, range = (0,1000))
plt.hist(lens_2)
plt.hist(lens_5)
plt.hist(lens_6)
plt.hist(lens_7)

