import pandas as pd   # pip install pandas

df = pd.read_excel('../stats/11/ЛИЦА.xls', sheet_name='ЛИЦА-1', index_col=0, comment='#')
print(df.to_string())
print(df.columns.ravel())