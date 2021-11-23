#%% Imports
import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

#%% Global initialization
pd.set_option('display.max_rows', 100)

data_file = '2019_headlines.pkl'
            
train_df = pd.DataFrame()
dev_df = pd.DataFrame()
test_df = pd.DataFrame()

#%% Path definitions
input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/03_FULL-final'
output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/04_full_final_2019'

#input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/00_raw/FULL'
#output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/02_full_dataset_1_year'

input_folder = '/home/sibanez/Projects/MyInvestor/NLP/01_spyproject/00_data/00_raw/01_full'
output_folder = '/home/sibanez/Projects/MyInvestor/NLP/01_spyproject/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/02_FULL_1year'

output_path_train = os.path.join(output_folder, 'model_train.pkl')
output_path_dev = os.path.join(output_folder, 'model_dev.pkl')
output_path_test = os.path.join(output_folder, 'model_test.pkl')

#%% Output dataset generation
input_path = os.path.join(input_folder, data_file)
input_df = pd.read_pickle(input_path)

train_df, dev_df = train_test_split(input_df, test_size=0.2, shuffle = False)

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
