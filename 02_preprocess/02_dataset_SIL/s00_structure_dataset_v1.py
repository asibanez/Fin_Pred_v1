# v0 -> Preprocess test datasets
# v1 -> Preprocess whole year datasets

#%% Imports
import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from datetime import datetime

#%% Global initialization
pd.set_option('display.max_rows', 100)

#%% Path definitions
news_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/2019/News'
trna_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/2019/TRNA'
output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/01_preprocessed'

news_paths = sorted(glob(os.path.join(news_folder, '*.txt')))
trna_news_paths = sorted(glob(os.path.join(trna_folder, '**/*News.txt.gz'), recursive = True))
trna_scores_paths = sorted(glob(os.path.join(trna_folder, '**/*Scores.txt.gz'), recursive = True))
output_pkl_path = os.path.join(output_folder, 'structured_2019.pkl')
output_csv_path = os.path.join(output_folder, 'structured_2019.csv')

#%% Load datasets
news_df = pd.DataFrame()
for path in tqdm(news_paths, desc = 'Loading news'):
    print(f'\n{datetime.now()} Loading file: {os.path.basename(path)}')
    aux_df = pd.read_json(path)
    print(f'{datetime.now()} Done')
    print(f'{datetime.now()} Normalizing')
    aux_df = pd.json_normalize(aux_df['Items'])
    print(f'{datetime.now()} Done')
    news_df = news_df.append(aux_df, ignore_index = True)
    
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
print(f'Size news_df = {news_df.shape}')
print(f'Size trna_news_df = {trna_news_df.shape}')
print(f'Size trna_scores_df = {trna_scores_df.shape}')

#%% Remove entries with more than one score
value_counts = pd.value_counts(trna_scores_df.id)
slicer = (value_counts < 2).values
ids_to_keep = list(value_counts[slicer].keys())
print(f'Number of entries to keep = {len(ids_to_keep):,}')

# Remove from news
print(f'{datetime.now()} Removing entries with > 1 scores from news')
slicer = news_df['data.id'].isin(ids_to_keep)
news_df = news_df[slicer].reset_index(drop = True)
print(f'{datetime.now()} Done')

# Remove from trna news
print(f'{datetime.now()} Removing entries with > 1 scores from trna_news')
slicer = trna_news_df['id'].isin(ids_to_keep)
trna_news_df = trna_news_df[slicer].reset_index(drop = True)
print(f'{datetime.now()} Done')

# Remove from trna scores
print(f'{datetime.now()} Removing entries with > 1 scores from trna_scores')
slicer = trna_scores_df['id'].isin(ids_to_keep)
trna_scores_df = trna_scores_df[slicer].reset_index(drop = True)
print(f'{datetime.now()} Done')

#%% Check dataset sizes
print(f'Size news_df = {news_df.shape}')
print(f'Size trna_news_df = {trna_news_df.shape}')
print(f'Size trna_scores_df = {trna_scores_df.shape}')

#%% Merge trna_news and trna_scores dataframes
trna_df = pd.merge(trna_news_df, trna_scores_df, how = 'inner', on = 'id')

#%% Check dataset sizes
print(f'Size news_df = {news_df.shape}')
print(f'Size trna_df = {trna_df.shape}')

#%% Checks - assetClass = COMPANY
print(pd.value_counts(trna_scores_df['assetClass']))


#%% Restructure trna dataframe
trna_df = trna_df[['id',
                  'amerTimestamp_x',
                  'headline',
                  'assetName',
                  'assetCodes',
                  'assetId',
                  'assetClass',
                  'sentimentClass',
                  'sentimentNegative',
                  'sentimentNeutral',
                  'sentimentPositive']]

#%% Check dataset sizes
print(f'Size trna = {trna_df.shape}')

#%% Restructure news dataframe
news_df_small = news_df.rename(columns = {'guid': 'id'})
news_df_small = news_df_small[['id',
                               'timestamps',
                               'data.body',
                               'data.messageType',
                               'data.mimeType']]

#%% Merge trna and news dataframes
data_df = pd.merge(trna_df, news_df_small, how = 'inner', on = 'id')

#%% Restructure output dataset
data_df = data_df.rename(columns = {'data.messageType': 'messageType',
                                    'data.mimeType': 'mimeType',
                                    'data.body': 'body'})
data_df = data_df[['id',
                   'amerTimestamp_x',
                   'assetCodes',
                   'assetId',
                   'assetName',
                   'assetClass',
                   'sentimentClass',
                   'sentimentNegative',
                   'sentimentNeutral',
                   'sentimentPositive',
                   'messageType',
                   'mimeType',
                   'headline',
                   'body']]

#%% Check dataset size
print(f'Size data = {data_df.shape}')

#%% Generate "difference" datasets
set_ids_trna = set(trna_df.id)
set_ids_news = set(news_df.guid)
set_difference = set_ids_trna.difference(set_ids_news)

diff_trna_df = trna_df[trna_df.id.isin(set_difference)]
intersec_trna_df = trna_df[~trna_df.id.isin(set_difference)]

diff_trna_news_df = trna_news_df[trna_news_df.id.isin(set_difference)]
intersec_trna_news_df = trna_news_df[~trna_news_df.id.isin(set_difference)]

#%% Check "difference" dataset sizes
print(f'Size trna_df = {trna_df.shape}')
print(f'Size news_df = {news_df.shape}')
print(f'Difference = {trna_df.shape[0] - news_df.shape[0]}\n')

print(f'Size diff_trna_df = {diff_trna_df.shape}')
print(f'Size intersec_trna_df = {intersec_trna_df.shape}\n')

print(f'Size diff_trna_news_df = {diff_trna_news_df.shape}')
print(f'Size intersec_trna_news_df = {intersec_trna_news_df.shape}\n')

#%% EDA
def counts_f(var):
    counts = pd.DataFrame(pd.value_counts(eval(var))).sort_index()
    counts.columns = ['totals']
    counts['%'] = [x / sum(counts.totals) for x in counts.totals]
    print('\n', '=' * 40, '\n')
    print(f'{var}\n{counts}')

#%%
vars_to_analyze = ['messageType']
for var in vars_to_analyze:
    counts_f('diff_trna_news_df.' + var)
    counts_f('intersec_trna_news_df.' + var)

#%% Save output datasets
data_df.to_pickle(output_pkl_path)
data_df.to_csv(output_csv_path)

#%% Save extra datasets
output_folder_extra = os.path.join(output_folder, '00_extra')

news_df_path = os.path.join(output_folder_extra, 'news_df_2019.pkl')
trna_news_df_path = os.path.join(output_folder_extra, 'trna_news_df_2019.pkl')
trna_scores_df_path = os.path.join(output_folder_extra, 'trna_scores_df_2019.pkl')
diff_trna_df_path = os.path.join(output_folder_extra, 'diff_trna_df_2019.pkl')
diff_trna_news_df_path = os.path.join(output_folder_extra, 'diff_trna_df_2019.pkl')
intersec_trna_df_path = os.path.join(output_folder_extra, 'intersec_trna_df_2019.pkl')
intersec_trna_news_df_path = os.path.join(output_folder_extra, 'intersec_trna_news_df_2019.pkl')

#%%
news_df.to_pickle(news_df_path)
trna_news_df.to_pickle(trna_news_df_path)
trna_scores_df.to_pickle(trna_scores_df_path)
diff_trna_df.to_pickle(diff_trna_df_path)
diff_trna_news_df.to_pickle(diff_trna_news_df_path)
intersec_trna_df.to_pickle(intersec_trna_df_path)
intersec_trna_news_df.to_pickle(intersec_trna_news_df_path)

