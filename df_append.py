import pandas as pd

df1 = pd.read_csv('../new_labeled_path_reports.csv', sep='|')
df2 = pd.read_csv('../new_labeled_rad_reports.csv', sep='|')

df1.append(df2)[['anon_id', 'text', 'label']].to_csv('../new_labeled_reports_full.csv', sep='|')
