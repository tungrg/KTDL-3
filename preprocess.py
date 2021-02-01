import numpy as np
import pandas as pd
raw_df = pd.read_csv('hawks.csv') #doc file hawks.csv
#rut trich so phut cot CaptureTime
minute = pd.to_numeric(raw_df['CaptureTime'].str.extract(r'[:.](\d+)', expand = False),errors = 'coerce')
#rut trich so gio cot CaptureTime
hour = pd.to_numeric(raw_df['CaptureTime'].str.extract(r'(\d+)[:.]', expand = False),errors = 'coerce')
#chuyen cot CaptureTime ve phut
raw_df['CaptureTime'] = (minute + 60*hour).fillna((minute + 60*hour).mean()).astype(int)
#Ham dien cac gia tri trong vao cac cot con lai
def fill_na_col(col):
    if  col.name in ['Wing', 'Weight', 'Culmen', 'Hallux', 'Tail', 'StandardTail', 'Tarsus','WingPitFat','KeelFat','Crop']:
        return col.fillna(round(col.mean(),2))
    return col
temp_df = raw_df.apply(fill_na_col)
raw_df = temp_df
raw_df.to_csv('test.csv', index = False) #luu ket qua xuong file test.csv (file này dùng để làm việc trong weka).