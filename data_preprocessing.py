
import pandas as pd


'Data Loading'
df_app = pd.read_csv('D:/Projects/Credit_Card_Approval/application_record.csv')

df_cre = pd.read_csv('D:/Projects/Credit_Card_Approval/credit_record.csv')


#print(df_app.isnull().sum())

#print(df_app.columns)

'Dropping column cause it has many null values'
df_app.drop('OCCUPATION_TYPE', axis = 1, inplace = True)

#print(df_app)

'Checking how many duplicates entries are there'
#print(len(df_app['ID']) - len(df_app['ID'].unique()))

'Removing duplicate entries'
df_app = df_app.drop_duplicates('ID', keep = 'last')

#print(df_app)

'Checking which columns are non-numerical'
df_cat = df_app.columns[(df_app.dtypes == 'object').values].tolist()

#print(df_cat)

'Checking which columns are numerical'
#print(df_app.columns[(df_app.dtypes != 'object').values].tolist())  

'Analysing count of values in non-numerical columns'
for v in df_app.columns[(df_app.dtypes == 'object').values].tolist():
    print(v, '\n')
    print(df_app[v].value_counts())
    print('**********************************************************')

'Analysing count of values in children count column'
#print(df_app['CNT_CHILDREN'].value_counts())

'Analysing count of values in children count column'
#print(df_app['DAYS_BIRTH'].value_counts())

'Changing column values day to years'
df_app['DAYS_BIRTH'] = round(df_app['DAYS_BIRTH']/-365, 0)

'Renaming column'
df_app.rename(columns={'DAYS_BIRTH':'AGE_YEARS'}, inplace = True)


'Checking column days employed values greater than 0'
#print(df_app[df_app['DAYS_EMPLOYED']>0]['DAYS_EMPLOYED'].unique())


#print(df_app['DAYS_EMPLOYED'].value_counts())

'Replacing value 365243 with 0'
df_app['DAYS_EMPLOYED'].replace(365243, 0, inplace= True)


'Converting values of days employed '
df_app['DAYS_EMPLOYED'] = abs(round(df_app['DAYS_EMPLOYED']/-365, 0))

'Renaming Column'
df_app.rename(columns={'DAYS_EMPLOYED':'YEARS_EMPLOYED'}, inplace= True)

'Analysing column'
#print(df_app['FLAG_MOBIL'].value_counts())
#print(df_app['FLAG_WORK_PHONE'].value_counts())
#print(df_app['FLAG_PHONE'].value_counts())
#print(df_app['FLAG_EMAIL'].value_counts())


'Droping columns'
df_app.drop(['FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL'], axis=1, inplace= True)

'Removing outliers'
upper_bound = df_app['CNT_CHILDREN'].quantile(0.999)
lower_bound = df_app['CNT_CHILDREN'].quantile(0.001) 


df_app = df_app[(df_app['CNT_CHILDREN']>=lower_bound) & (df_app['CNT_CHILDREN']<=upper_bound)]


upper_bound = df_app['AMT_INCOME_TOTAL'].quantile(0.999)
lower_bound = df_app['AMT_INCOME_TOTAL'].quantile(0.001)

df_app = df_app[(df_app['AMT_INCOME_TOTAL']>=lower_bound) & (df_app['AMT_INCOME_TOTAL']<=upper_bound)]



upper_bound = df_app['YEARS_EMPLOYED'].quantile(0.999)
lower_bound = df_app['YEARS_EMPLOYED'].quantile(0.001)


df_app = df_app[(df_app['YEARS_EMPLOYED']>=lower_bound) & (df_app['YEARS_EMPLOYED']<=upper_bound)]


upper_bound = df_app['CNT_FAM_MEMBERS'].quantile(0.999)
lower_bound = df_app['CNT_FAM_MEMBERS'].quantile(0.001)

df_app = df_app[(df_app['CNT_FAM_MEMBERS']>=lower_bound) & (df_app['CNT_FAM_MEMBERS']<=upper_bound)]



'Wokring on credit record file'
'Checking any null values are there or not'
#print(df_cre.isnull().sum())


'Checking status column values count'
#print(df_cre['STATUS'].value_counts())






'0 : good client and 1: bad client'
df_cre['STATUS'].replace(['X', 'C'], 0, inplace=True)

df_cre['STATUS'].replace(['2', '3', '4', '5'], 1, inplace=True)

df_cre['STATUS'] = df_cre['STATUS'].astype('int')

#df_cre['STATUS'].value_counts(normalize=True)*100


'removing multiple entries in ID Column'
df_cre_tran = df_cre.groupby('ID').agg('max').reset_index()

'Droping column months balance'
df_cre_tran.drop('MONTHS_BALANCE', axis = 1, inplace = True)

#print(df_cre_tran.head())

df_cre_tran['STATUS'].value_counts(normalize=True)*100

'joining two dataframes using common column / merging two dataframes'
main_df = pd.merge(df_app, df_cre_tran, on='ID', how='inner')

main_df.drop('ID', axis=1, inplace=True)

#print(len(main_df) - len(main_df).drop_duplicates())

main_df = main_df.drop_duplicates()


main_df.to_csv('D:/Projects/Credit_Card_Approval/clean.csv', index=False)







































