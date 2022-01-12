# v0 -> Works with Multiple year files: train.pkl / validation.pkl / test.pkl

#%% Imports
import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

#%% Global initialization
pd.set_option('display.max_rows', 100)
data_file = 'validation.pkl' #train.pkl / validation.pkl / test.pkl
year = 2017
shuffle = False
            
train_df = pd.DataFrame()
dev_df = pd.DataFrame()

#%% Path definitions

# Laptop
input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/03_FULL-MA-final'
output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/07_2017_MA_final_sorted'

# Server
#input_folder = '/home/sibanez/Projects/MyInvestor/NLP/01_spyproject/00_data/00_raw/01_full'
#output_folder = '/home/sibanez/Projects/MyInvestor/NLP/01_spyproject/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/02_FULL_1year'

input_path = os.path.join(input_folder, data_file)
output_path_train = os.path.join(output_folder, 'model_train.pkl')
output_path_dev = os.path.join(output_folder, 'model_dev.pkl')

#%% Dataset reading and slicing
data_df = pd.read_pickle(input_path)
slicer = [x.year == year for x in data_df.date]
output_df = data_df[slicer]

#%% Check dataset size and years present
print(f'Shape dataset before slicing = {data_df.shape}')
print(f'Shape dataset after slicing = {output_df.shape}\n')

#%% Check valuecounts for years
years = [x.year for x in output_df.date]
print(pd.value_counts(years))

#%% Dataset split
train_df, dev_df = train_test_split(output_df, test_size=0.2, shuffle = shuffle)

#%% Check dataset size
print(f'Shape train set = {train_df.shape}')
print(f'Shape dev set = {dev_df.shape}')

print(f'% train set = {len(train_df) / len(output_df):.2f}')
print(f'% dev set = {len(dev_df) / len(output_df):.2f}')

#%% Save output datasets
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

print(f'{datetime.now()} Saving datasets')
train_df.to_pickle(output_path_train)
dev_df.to_pickle(output_path_dev)
print(f'{datetime.now()} Done')
