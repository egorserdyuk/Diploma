import pandas as pd   # pip install pandas

df = pd.read_excel('../stats/11/ЛИЦА.xls', sheet_name='ЛИЦА-1', index_col=0)
df = df.reset_index()
df = df.rename(columns=lambda x: x if not 'Unnamed' in str(x) else None)
df = df.dropna(how='all')
df.columns = pd.Series(df.columns).fillna(method='ffill', axis=0)
print(df.to_string())
print(df.columns)

