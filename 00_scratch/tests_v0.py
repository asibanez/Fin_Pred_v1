import os
import pandas as pd
from glob import glob

path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/03_spy_project/00_data/2019/TRNA'

len(glob(os.path.join(path,'**/*.gz'), recursive=True))


path_test = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/03_spy_project/00_data/2019/TRNA/06/01/NA_ENT0.TR.News.CMPNY_AMER.EN.2019-06-01_00.41020104.Flat-News.txt.gz'

data = pd.read_table(path_test, compression='infer')
