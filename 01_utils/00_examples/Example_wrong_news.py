#%% Imports
import os
import glob
import pandas as pd

#%% Global initialization
pd.set_option('display.max_rows', 100)

#%% Path definitions
news_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/03_spy_project/00_data/00_testing/News'
trna_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/03_spy_project/00_data/00_testing/TRNA'

news_path = os.path.join(news_folder, 'News.RTRS_CMPNY_AMER.202006.0214.txt')
trna_news_paths = glob.glob(os.path.join(trna_folder, '*News.txt'))
trna_scores_paths = glob.glob(os.path.join(trna_folder, '*Scores.txt'))

#%% Load datasets
news_df = pd.json_normalize(pd.read_json(news_path)['Items'])

trna_news_df = pd.DataFrame()
for path in trna_news_paths:
    aux_df = pd.read_table(path) 
    trna_news_df = trna_news_df.append(aux_df, ignore_index = True)
    
trna_scores_df = pd.DataFrame()
for path in trna_scores_paths:
    aux_df = pd.read_table(path) 
    trna_scores_df = trna_scores_df.append(aux_df, ignore_index = True)

#%% Check dataset sizes
print(f'Size news = {news_df.shape}')
print(f'Size trna_news = {trna_news_df.shape}')
print(f'Size trna_scores = {trna_scores_df.shape}')

#%% Compute # of scores per news
pd.value_counts(trna_scores_df.id)[0:100]

#%% Visualize scores and news text
id_trna = 'tr:L1N2DE0VF_2006012mLFCIVAZDLLkzEZvABg60AHoLqyzF2eZ7P67b'
id_news = id_trna.split(':')[1]

trna_news_df[trna_news_df.id == id_trna]
trna_news_df.loc[1602]

trna_scores_df[trna_scores_df.id == id_trna]
trna_scores_df.loc[1791]
trna_scores_df.loc[1792]
trna_scores_df.loc[1793]

news_df[news_df['data.id'] == id_news]
news_df.loc[5288]

#%% Print headline and body
body = news_df.loc[5288]['data.body']
headline = news_df.loc[5288]['data.headline']
print(f'HEADLINE =====================\n\n{headline}\n')
print(f'BODY =====================\n\n{body}\n')
