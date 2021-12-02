# v0 -> Generates train, dev, test sets from single years

#%% Imports
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

#%% Path definition

# Local
input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/04_FULL-MA-final_fixed/00_single_years'
output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/04_FULL-MA-final_fixed/01_train_dev_test'

# Server
#input_folder = '/home/sibanez/Projects/00_MyInvestor/00_data/00_raw/04_FULL-MA-final-fixed/2018_headlines.pkl'
#output_folder = '/home/sibanez/Projects/00_MyInvestor/00_data/02_preprocessed/04_BOW/00_binary/01_2018_no_imbalance'

output_path_train = os.path.join(output_folder, 'model_train.pkl')
output_path_dev = os.path.join(output_folder, 'model_dev.pkl')
output_path_test = os.path.join(output_folder, 'model_test.pkl')

#%% Global initialization
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

#%% Generate and save train_set
aux_df = pd.DataFrame()
for file in tqdm(train_files, desc = 'Generating train set'):
    path = os.path.join(input_folder, file)
    data_df = pd.read_pickle(path)
    aux_df = aux_df.append(data_df)

print(f'\nShape train set = {aux_df.shape}')
years = [x.year for x in aux_df.date]
print(f'\nValue counts years train:\n{pd.value_counts(years).sort_index()}')
    
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

print(f'\n{datetime.now()}: Saving train set')
aux_df.to_pickle(output_path_train)
print(f'{datetime.now()}: Done')

#%% Generate and save dev set
aux_df = pd.DataFrame()
for file in tqdm(dev_files, desc = 'Generating dev set'):
    path = os.path.join(input_folder, file)
    data_df = pd.read_pickle(path)
    aux_df = aux_df.append(data_df)

print(f'\nShape dev set = {aux_df.shape}')
years = [x.year for x in aux_df.date]
print(f'\nValue counts years dev:\n{pd.value_counts(years).sort_index()}')

print(f'\n{datetime.now()}: Saving dev set')
aux_df.to_pickle(output_path_dev)
print(f'{datetime.now()}: Done')

#%% Generate and save test set
aux_df = pd.DataFrame()
for file in tqdm(test_files, desc = 'Generating test set'):
    path = os.path.join(input_folder, file)
    data_df = pd.read_pickle(path)
    aux_df = aux_df.append(data_df)

print(f'\nShape test set = {aux_df.shape}')
years = [x.year for x in aux_df.date]
print(f'\nValue counts years test:\n{pd.value_counts(years).sort_index()}')

print(f'\n{datetime.now()}: Saving test set')
aux_df.to_pickle(output_path_test)
print(f'{datetime.now()}: Done')