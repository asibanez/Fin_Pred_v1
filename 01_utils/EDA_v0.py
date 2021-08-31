#%% Imports
import os
import json
import glob
import tqdm
import codecs
import chardet
import pandas as pd
import matplotlib.pyplot as plt

#%% Set display options
pd.set_option('display.max_rows', 100)

#%% Define paths
# Inputs
data_folder = 'C:\\Users\\siban\\Dropbox\\BICTOP\\MyInvestor\\06_model\\02_datasets\\01_sample_data'
mr_news_folder = os.path.join(data_folder, 'Machine Readabele News Sample - Option 1\\TRNA All Companies English 08-28-2019 v41')
newsfeed_path = os.path.join(data_folder, 'Reuters News feed - Option 3\\Reuters Americas Company News 2019-08-28.txt')

# Outputs
output_df_path = os.path.join(data_folder, 'output_df.txt')
output_main_df_path = os.path.join(data_folder, 'output_main_df.txt')
output_summary_path = os.path.join(data_folder, 'output_summary.txt')

#%% Detect file encoding
#enc=chardet.detect(open(newsfeed_path,'rb').read())['encoding']

#%% Read news feed (Option 3)
with codecs.open(newsfeed_path, 'r', 'utf-8') as fr:
    newsfeed = json.load(fr)

# Reformat json into dataframe
newsfeed = pd.json_normalize(newsfeed,['Items'])
newsfeed = newsfeed.rename(columns = {'guid':'id'})
newsfeed['data.body'] = [x.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ') for x in newsfeed['data.body']]

#%% Read mr_news
file_list = glob.glob(os.path.join(mr_news_folder,'*News.txt'))

for idx, fr in tqdm.tqdm(enumerate(file_list), total = len(file_list)):
    aux_df = pd.read_csv(fr, delimiter = '\t')
    if idx == 0:
        mr_news_df = aux_df
    else:
        mr_news_df = pd.concat([mr_news_df, aux_df])

mr_news_df.id = [x.strip ('tr:') for x in mr_news_df.id]

#%% Read mr_news_scores
file_list = glob.glob(os.path.join(mr_news_folder,'*Scores.txt'))

for idx, fr in tqdm.tqdm(enumerate(file_list), total = len(file_list)):
    aux_df = pd.read_csv(fr, delimiter = '\t')
    if idx == 0:
        mr_news_scores_df = aux_df
    else:
        mr_news_scores_df = pd.concat([mr_news_scores_df, aux_df])

mr_news_scores_df.id = [x.strip ('tr:') for x in mr_news_scores_df.id]

#%% Check duplicated columns in mr_news_df and mr_news_scores_df
dup_cols = sorted(list(set(mr_news_df.columns).intersection (set(mr_news_scores_df.columns))))
dup_cols = [x for x in dup_cols if x != 'id']
mr_news_scores_df = mr_news_scores_df.drop(columns = dup_cols)

#%% Merge mr_news dataframes
mr_news_all_df = mr_news_df.merge(mr_news_scores_df, how = 'inner', on = 'id')

#%% Check dimensions for merging dataframes
unique_ids_mr_news = set(mr_news_all_df.id)
num_ids_mr_news = len(list(unique_ids_mr_news))

unique_ids_newsfeed = set(newsfeed.id)
num_ids_newsfeed = len(list(unique_ids_newsfeed))

ids_both = unique_ids_mr_news.intersection(unique_ids_newsfeed)
num_ids_both = len(list(ids_both))

print('# ids in mr_news', num_ids_mr_news, '\n',
      '# ids in newsfeed', num_ids_newsfeed, '\n',
      '# ids in both', num_ids_both, '\n')                  

#%% Merge mr_news dataframe & newsfeed dataframe 
output_df = mr_news_all_df.merge(newsfeed, how = 'inner', on = 'id')
output_main_df = output_df[['id', 'amerTimestamp', 'headline', 'assetClass',
                            'assetName', 'sentimentClass', 'data.body']]

#%% Show relevant information for specific idx
idx_range = (0,10)

for idx in range(idx_range[0], idx_range[1]):
    print('* ID: ', output_main_df.iloc[idx].id, '\n' +
          '* HEADLINE: ', output_main_df.iloc[idx].headline, '\n' +
          '* COMPANY: ', output_main_df.iloc[idx].assetName, '\n' +
          '* TEXT: ', output_main_df.iloc[idx]['data.body'].replace('\n', ' '), '\n' +
          '* SENTIMENT: ', output_main_df.iloc[idx].sentimentClass)
 
    print('\n', '-' * 100, '\n')

#%% Plot histogram of token lengths
text_len = [len(x.split(' ')) for x in output_main_df['data.body']]

figure = plt.figure(1)
plt.hist(text_len)
plt.xlabel('# tokens')
plt.ylabel('freq')
plt.show()

#%% Compute number of news longer than n tokens
n_tokens = 512
lens_long = [x for x in text_len if x > n_tokens]
print(f'Number of news with more than {n_tokens} tokens = {len(lens_long)}')

#%% Save output dataframe
output_df.to_csv(output_df_path, sep = ',', line_terminator = '\n', encoding = 'utf-8')   
output_main_df.to_csv(output_main_df_path, sep = ',', line_terminator = '\n', encoding = 'utf-8')

#%% Save summary txt
with codecs.open(output_summary_path, 'w', 'utf-8') as fw:
    for idx in range(len(output_main_df)):
        fw.write('* News #:' + str(idx) + '\n' +
                 '* ID: ' + output_main_df.iloc[idx].id + '\n' +
                 '* HEADLINE: ' + output_main_df.iloc[idx].headline + '\n' +
                 '* COMPANY: ' + output_main_df.iloc[idx].assetName + '\n' +
                 '* TEXT: ' + output_main_df.iloc[idx]['data.body'].replace('\n', ' ') + '\n' +
                 '* SENTIMENT: ' + str(output_main_df.iloc[idx].sentimentClass))
        fw.write('\n' + '-' * 100 + '\n')
        