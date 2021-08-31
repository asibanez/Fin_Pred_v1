import os
import json
import codecs
import pandas as pd

path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/03_spy_project/00_data/News.RTRS_CMPNY_AMER.202007.0214.txt'

df = pd.read_json(path)

fr = codecs.open(path, 'r', 'utf-8')
data = json.load(fr)

#%%
dict1 = {'school': ['maristas', 'salesianos'],
         'location': ['vigo', 'barcelona'],
         'ranking': [1 ,2]}
#%%
dict2 = [{'school': 'maristas',
          'location': 'vigo',
          'ranking': 1},
         {'school': 'salesianos',
          'location': 'barcelona',
          'ranking': 2}]
#%%

kiki = pd.json_normalize(dict1)
kuku = pd.json_normalize(dict2)

#%%
dict3 = {'desc': 'database',
         'Items':[{'Id': 1,
                   'Name': 'Kiki'},
                  {'Id': 2,
                   'Name': 'Kuku'}]}
#%%
dict4 = {'desc': 'database',
         'Items':[{'Id': 1,
                   'Name': 'Kiki'},
                  {'Id': 2,
                   'Name': 'Kuku'}],
         }
#%%

test = pd.json_normalize(dict4, 'Items', ['desc'])
test = pd.json_normalize(dict4, 'Items')

#%%


