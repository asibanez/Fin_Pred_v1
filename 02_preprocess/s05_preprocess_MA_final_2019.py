#%% Imports
import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

#%% Global initialization
pd.set_option('display.max_rows', 100)
data_file = '2019_headlines.pkl'
shuffle = False
            
train_df = pd.DataFrame()
dev_df = pd.DataFrame()
test_df = pd.DataFrame()

#%% Path definitions
# Laptop
input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/03_FULL-MA-final'
output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/05_2019_MA_final_sorted'
# Server
#input_folder = '/home/sibanez/Projects/MyInvestor/NLP/01_spyproject/00_data/00_raw/01_full'
#output_folder = '/home/sibanez/Projects/MyInvestor/NLP/01_spyproject/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/02_FULL_1year'

output_path_train = os.path.join(output_folder, 'model_train.pkl')
output_path_dev = os.path.join(output_folder, 'model_dev.pkl')

#%% Output dataset generation
input_path = os.path.join(input_folder, data_file)
input_df = pd.read_pickle(input_path)

train_df, dev_df = train_test_split(input_df, test_size=0.2, shuffle = shuffle)

#%% Check dataset size
print(f'Shape train set = {train_df.shape}')
print(f'Shape dev set = {dev_df.shape}')

print(f'% train set = {len(train_df) / len(input_df):.2f}')
print(f'% dev set = {len(dev_df) / len(input_df):.2f}')

#%% Save output datasets
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

print(f'{datetime.now()} Saving datasets')
train_df.to_pickle(output_path_train)
dev_df.to_pickle(output_path_dev)
print(f'{datetime.now()} Done')
