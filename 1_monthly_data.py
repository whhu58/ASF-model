import os
import pandas as pd 
from tqdm import tqdm


# ------------------- processing commits ---------------------- 
print('reading commits...')
df = pd.read_csv('./commits_final.csv')
print('grouping by project...')
df = dict(tuple(df.groupby(df['project_name'])))
to_path = './monthly_data/commits/'
if not os.path.exists(to_path):
	os.makedirs(to_path)

print('grouping by period...')
for project in tqdm(df):
	monthly_df_dict = dict(tuple(df[project].groupby(df[project]['month'])))
	for month in monthly_df_dict:
		monthly_df = monthly_df_dict[month]
		monthly_df = monthly_df[monthly_df['dealised_author_full_name'].notna()]
		if monthly_df.empty: continue
		file_path = to_path + '{}__{}.csv'.format(project, str(int(month)))
		monthly_df.to_csv(file_path, index=False)
print('Commits Done.')

# ------------------- processing emails ---------------------- 
print('reading emails...')
df = pd.read_csv('./emails_final.csv')
print('grouping by project...')
df = dict(tuple(df.groupby(df['project_name'])))
to_path = './monthly_data/emails/'
if not os.path.exists(to_path):
	os.makedirs(to_path)

print('grouping by period...')
for project in tqdm(df):
	monthly_df_dict = dict(tuple(df[project].groupby(df[project]['month'])))
	for month in monthly_df_dict:
		monthly_df = monthly_df_dict[month]
		monthly_df = monthly_df[monthly_df['dealised_author_full_name'].notna()]
		if monthly_df.empty: continue
		file_path = to_path + '{}__{}.csv'.format(project, str(int(month)))
		monthly_df.to_csv(file_path, index=False)
print('Emails Done.')
