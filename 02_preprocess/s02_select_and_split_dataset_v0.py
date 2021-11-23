# v0 Splits in train, dev and test sets

#%% Imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split

#%% Global initialization
pd.set_option('display.max_rows', 100)

#%% Path definitions
input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/00_toy'
#input_folder = '/data/rsg/nlp/sibanez/00_temp/01_fin_pred/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary'
output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/05_toy_TEST'

input_path = os.path.join(input_folder, 'preproc_2019_mapped_SP500_full.pkl')
output_path_train = os.path.join(output_folder, 'model_train.pkl')
output_path_dev = os.path.join(output_folder, 'model_dev.pkl')

#%% Load dataset
dataset_df = pd.read_pickle(input_path)

#%% Slice, remove invalid values and rename columns
X_values = ['token_ids_headline',
            'token_types_headline',
            'att_masks_headline']
#%%
Y_value = 'labels_p1'
output_df = dataset_df[X_values + [Y_value]]
slicer = dataset_df[Y_value] == 'NaN'
output_df = output_df[~slicer]
output_df.columns = ['token_ids',
                     'token_types',
                     'att_masks',
                     'Y']

#%% Check dataset size
print(f'Shape dataset before slicing = {dataset_df.shape}')
print(f'Shape dataset after slicing = {output_df.shape}')

#%% Split datasets
train_set_df, test_dev_set_df = train_test_split(output_df, test_size = 0.2, shuffle = True)
dev_set_df, test_set_df = train_test_split(output_df, test_size = 0.5, shuffle = True)

#%% Check dataset sizes
print(f'\nShape train set = {train_set_df.shape}')
print(f'% train set: {len(train_set_df)/len(output_df)*100:.2f}%\n')
print(f'Shape dev set = {dev_set_df.shape}')
print(f'Dev: {len(dev_set_df)/len(output_df)*100:.2f}%\n')
print(f'Shape test set = {test_set_df.shape}')
print(f'Test: {len(test_set_df)/len(output_df)*100:.2f}%\n')

#%% Save datasets
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)
train_set_df.to_pickle(output_path_train)
dev_set_df.to_pickle(output_path_dev)
