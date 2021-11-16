#%% Imports
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

#%% Global initialization
pd.set_option('display.max_rows', 100)

train_files = ['1996_headlines.pkl',
			   '1997_headlines.pkl',
			   '1998_headlines.pkl',
			   '1999_headlines.pkl',
			   '2000_headlines.pkl',
			   '2001_headlines.pkl',
			   '2002_headlines.pkl',
			   '2003_headlines.pkl',
			   '2004_headlines.pkl',
			   '2005_headlines.pkl',
			   '2006_headlines.pkl',
			   '2007_headlines.pkl',
			   '2008_headlines.pkl',
			   '2009_headlines.pkl',
			   '2010_headlines.pkl',
			   '2011_headlines.pkl',
			   '2012_headlines.pkl',
               '2013_headlines.pkl',
               '2014_headlines.pkl']

dev_files = ['2015_headlines.pkl',
             '2016_headlines.pkl',
             '2017_headlines.pkl',
             '2018_headlines.pkl']

test_files = ['2019_headlines.pkl',
              '2020_headlines.pkl',
              '2021_headlines.pkl']

train_df = pd.DataFrame()
dev_df = pd.DataFrame()
test_df = pd.DataFrame()

#%% Path definitions
#input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/FULL'
#output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/01_full_dataset'

input_folder = '/home/sibanez/Projects/MyInvestor/NLP/01_spyproject/00_data/00_raw/01_full'
output_folder = '/home/sibanez/Projects/MyInvestor/NLP/01_spyproject/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/01_FULL'

output_path_train = os.path.join(output_folder, 'model_train.pkl')
output_path_dev = os.path.join(output_folder, 'model_dev.pkl')
output_path_test = os.path.join(output_folder, 'model_test.pkl')

#%% Output dataset generation
for file in tqdm(train_files, desc = 'Train set'):
    input_path = os.path.join(input_folder, file)
    aux_df = pd.read_pickle(input_path)
    train_df = train_df.append(aux_df)

for file in tqdm(dev_files, desc = 'Dev set'):
    input_path = os.path.join(input_folder, file)
    aux_df = pd.read_pickle(input_path)
    dev_df = dev_df.append(aux_df)

for file in tqdm(test_files, desc = 'Test set'):
    input_path = os.path.join(input_folder, file)
    aux_df = pd.read_pickle(input_path)
    test_df = test_df.append(aux_df)

#%% Check dataset size
len_full_set = len(train_df) + len(dev_df) + len (test_df)

print(f'Shape train set = {train_df.shape}')
print(f'Shape dev set = {dev_df.shape}')
print(f'Shape test set = {test_df.shape}')

print(f'% train set = {len(train_df) / len_full_set:.2f}')
print(f'% dev set = {len(dev_df) / len_full_set:.2f}')
print(f'% test set = {len(test_df) / len_full_set:.2f}')

#%% Save output datasets
print(f'{datetime.now()} Saving datasets')
train_df.to_pickle(output_path_train)
dev_df.to_pickle(output_path_dev)
test_df.to_pickle(output_path_test)
print(f'{datetime.now()} Done')
