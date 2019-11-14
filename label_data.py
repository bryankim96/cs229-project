import pandas as pd

# Conditions for a report indicating a positive response
def good_results(results):
	if " yp" in results and "T0" in results and "N0" in results:
		return 1
	elif " yp" in results:
		return -1
    else return 0

# Conditions for a report corresponding to a patient with a future positive response
def good_results_exist(target_id, target_time, row_id, row_time, response):
    target_day, target_month, target_year = target_time.split('-')
    row_day, row_month, row_year = row_time.split('-')
    later_date = int(row_year) > int(target_year) or (int(row_year) == int(target_year) and (int(row_month) > int(target_month) or (int(row_month) == int(target_month) and int(row_day) > int(target_day))))
    return target_id == row_id and later_date and response == 1

# Conditions for a report corresponding to a patient with a future negative response
def bad_results_exist(target_id, target_time, row_id, row_time, response):
    target_day, target_month, target_year = target_time.split('-')
    row_day, row_month, row_year = row_time.split('-')
    later_date = int(row_year) > int(target_year) or (int(row_year) == int(target_year) and (int(row_month) > int(target_month) or (int(row_month) == int(target_month) and int(row_day) > int(target_day))))
    return target_id == row_id and later_date and response == -1

# Labels whether patient had positive response at some point in the future
def label(row_id, row_time, df):
    if True in df.apply(lambda row: good_results_exist(row_id, row_time, row['ANON_ID'], row['RESULT_TIME'], row['GOOD_RESPONSE']), axis = 1).values:
        return 1
    elif True in df.apply(lambda row: bad_results_exist(row_id, row_time, row['ANON_ID'], row['RESULT_TIME'], row['GOOD_RESPONSE']), axis = 1).values:
        return -1
    else:
        return 0

def label_data(read_path, write_path):
    df = pd.read_csv(read_path, sep='|')

    response_df = df.copy()

    response_df['GOOD_RESPONSE'] = response_df.apply(lambda row: good_results(row['REPORT']), axis = 1)

    label_df = response_df.copy()

    label_df['LABEL'] = label_df.apply(lambda row: label(row['ANON_ID'], row['RESULT_TIME'], response_df), axis = 1)

    print(label_df.head(5))

    label_df.to_csv(write_path)

if __name__ == "__main__":
    label_data('../haruka_pathology_reports_111618.csv', '../labeled_pathology_reports.csv')
    # label_data('test_data.csv', 'labeled_test_data.csv')

