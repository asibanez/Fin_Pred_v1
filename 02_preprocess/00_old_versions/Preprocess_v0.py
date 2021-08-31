# v0 -> Preprocess test datasets

#%% Imports
import os
import glob
import pandas as pd
from tqdm import tqdm

#%% Global initialization
pd.set_option('display.max_rows', 100)

#%% Path definitions
news_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/03_spy_project/00_data/00_testing/News'
trna_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/03_spy_project/00_data/00_testing/TRNA'

news_path = os.path.join(news_folder, 'News.RTRS_CMPNY_AMER.202006.0214.txt')
trna_news_paths = glob.glob(os.path.join(trna_folder, '*News.txt'))
trna_scores_paths = glob.glob(os.path.join(trna_folder, '*Scores.txt'))

#%% Load datasets
news_df = pd.read_json(news_path)
news_df = pd.json_normalize(news_df['Items'])

trna_news_df = pd.DataFrame()
for path in tqdm(trna_news_paths, desc = 'Loading trna news'):
    aux_df = pd.read_table(path) 
    trna_news_df = trna_news_df.append(aux_df, ignore_index = True)
    
trna_scores_df = pd.DataFrame()
for path in tqdm(trna_scores_paths, desc = 'Loading trna scores'):
    aux_df = pd.read_table(path) 
    trna_scores_df = trna_scores_df.append(aux_df, ignore_index = True)

#%% Process datasets for consistency
trna_news_df.id = [x.split(':')[1] for x in trna_news_df.id]
trna_scores_df.id = [x.split(':')[1] for x in trna_scores_df.id]

#%% Check dataset sizes
print(f'Size news = {news_df.shape}')
print(f'Size trna_news = {trna_news_df.shape}')
print(f'Size trna_scores = {trna_scores_df.shape}')

#%% Remove entries with more than one score
value_counts = pd.value_counts(trna_scores_df.id)
slicer = (value_counts < 2).values
ids_to_keep = list(value_counts[slicer].keys())
print(f'Number of entries to keep = {len(ids_to_keep):,}')

# Remove from news
slicer = news_df['data.id'].isin(ids_to_keep)
news_df = news_df[slicer].reset_index(drop = True)

# Remove from trna news
slicer = trna_news_df['id'].isin(ids_to_keep)
trna_news_df = trna_news_df[slicer].reset_index(drop = True)

# Remove from trna scores
slicer = trna_scores_df['id'].isin(ids_to_keep)
trna_scores_df = trna_scores_df[slicer].reset_index(drop = True)

#%% Check dataset sizes
print(f'Size news = {news_df.shape}')
print(f'Size trna_news = {trna_news_df.shape}')
print(f'Size trna_scores = {trna_scores_df.shape}')

#%% Merge trna_news and trna_scores dataframes
trna_df = pd.merge(trna_news_df, trna_scores_df, how = 'inner', on = 'id')

#%% Check dataset sizes
print(f'Size news = {news_df.shape}')
print(f'Size trna = {trna_df.shape}')

#%% Process trna dataframe
trna_df = trna_df[['id',
                  'amerTimestamp_x',
                  'headline',
                  'assetName',
                  'assetClass',
                  'sentimentClass',
                  'sentimentNegative',
                  'sentimentNeutral',
                  'sentimentPositive']]

#%% Check dataset sizes
print(f'Size trna = {trna_df.shape}')

#%% Process news dataframe
news_df = news_df.rename(columns = {'guid': 'id'})
news_df = news_df[['id',
                   'timestamps',
                   'data.body',
                   'data.messageType',
                   'data.mimeType']]

#%% Merge trna and news dataframes
data_df = pd.merge(trna_df, news_df, how = 'inner', on = 'id')

#%% Check dataset sizes
print(f'Size data = {data_df.shape}')
