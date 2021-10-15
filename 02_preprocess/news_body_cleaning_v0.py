# Imports
import re
import pandas as pd
from tqdm import tqdm

#%% Path definition
path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/01_structured/structured_2019_mapped_SP500_full_FULL.pkl'

#%% Data loading and filtering of empty bodies
data = pd.read_pickle(path)
slicer = data['body'] != ''
data_filt = data[slicer]

#%% Check size datasets
print(f'len(bodies) = {data.shape[0]}')
print(f'len(bodies_clean) = {data_filt.shape[0]}')

#%% Regex definition - parts to remove
regex_list = ['http.*',
              'Source:.*',
              '- Note:.*',
              '<^[\s\S]*^>',
              #'^[\s\S]* - ',
              'Short Link:.*',
              'Source text.*',
              #'Description:.*',
              '\(\([\s\S]*\)\)',
              '- Source link:.*',
              '\(Compiled[\s\S]*\)',
              '\(Writing[\s\S]*\)',
              '\(Reporting[\s\S]*\)',
              '\(Additional reporting[\s\S]*\)',
              'Further company coverage:.*',
              'Verified transcript not available',
              'Click the following link.*',
              '\S*@\S*\s?']
              #CHECK 'Video Transcript:' ?????

#%% News body processing
body_clean = []
for body in tqdm(data_filt['body']):
    for regex in regex_list:
        body = re.sub(regex, '', body)
    body_clean.append(body)

#%% Check - print single bodies
pos = 500
print(data_filt.iloc[pos]['body'])
print(f'\n{"="*40}\n')
print(body_clean[pos])
   
#%% Check - # of URLS
url_in_body = ['http' in x for x in data_filt['body']]
url_in_body_clean = ['http' in x for x in body_clean]
print(f'# of urls in body = {sum(url_in_body)}')
print(f'# of urls in body_clean = {sum(url_in_body_clean)}')

#%% Check - # of emails
email_in_body = ['@' in x for x in data_filt['body']]
email_in_body_clean = ['@' in x for x in body_clean]
print(f'# of emails in body = {sum(email_in_body)}')
print(f'# of emails in body_clean = {sum(email_in_body_clean)}')

#%% Check - print range of bodies
beg = 150
end = 160
for idx in range(beg, end):
    print(f'idx = {idx}\n')
    print (data_filt['body'].iloc[idx])
    print(f'\n{"=" * 40} XXXXXX {"=" * 40}\n')
