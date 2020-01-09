import pandas as pd


# Conditions for a report indicating a positive response
def good_results_tumor(trneg, res, tcode, ncode, dtcode, dncode):
    if trneg == 0.0 and (res == 10.0 or (dtcode == 0.0 and dncode == 0.0) or (tcode == 'p0' and 'p0' in ncode)):
        return 1
    elif trneg == 0.0 and res in [20.0, 30.0]:
        return 0
    elif trneg == 0.0:
        return 2
    else:
        return -1


def good_results_comorbid(time):
    if time >= 10:
        return 1
    elif time < 2:
        return 0
    else:
        return -1


def determine_response(row):
    if row['RESPONSE1'] == 1 or row['RESPONSE2'] == 1:
        return 1
    elif row['RESPONSE1'] == 0 or row['RESPONSE2'] == 0:
        return 0
    else:
        return -1


# Conditions for a report corresponding to a patient with a future positive response
def good_results_exist(target_id, row_id, response):
    return target_id == row_id and response == 1


# Conditions for a report corresponding to a patient with a future negative response
def bad_results_exist(target_id, row_id, response):
    return target_id == row_id and response == 0


# Labels whether patient had positive response at some point in the future
def label(row_id, df):
    print(row_id)
    if True in df.apply(lambda row: good_results_exist(row_id, row['ANON_ID'], row['RESPONSE']), axis=1).values:
        return 1
    elif True in df.apply(lambda row: bad_results_exist(row_id, row['ANON_ID'], row['RESPONSE']), axis=1).values:
        return 0
    else:
        return -1


def label_data(read_path_1, read_path_2, read_path_3, write_path):
    df1 = pd.read_csv(read_path_1, sep='|')
    df1.sort_values(by=['ANON_ID'])
    df2 = pd.read_csv(read_path_2, sep=',')
    df3 = pd.read_csv(read_path_3, sep='|')

    response_df1 = pd.DataFrame(columns=['ANON_ID', 'RESPONSE1'])

    response_df1['ANON_ID'] = df2['ANON_ID']

    response_df1['RESPONSE1'] = df2.apply(
        lambda row: good_results_tumor(row['CS_SITE_SPEC_F16'], row['CS_SITE_SPEC_F21'], row['TCODE_P'], row['NCODE_P'],
                                       row['DERIVEDAJCC7T'], row['DERIVEDAJCC7N']), axis=1)

    keep_response1 = response_df1['RESPONSE1'] != -1
    response_df1 = response_df1[keep_response1]

    response_df2 = pd.DataFrame(columns=['ANON_ID', 'RESPONSE2'])

    response_df2['ANON_ID'] = df3['ANON_ID']

    response_df2['RESPONSE2'] = df3.apply(lambda row: good_results_comorbid(row['SURV_TIME']), axis=1)

    keep_response2 = response_df2['RESPONSE2'] != -1
    response_df2 = response_df2[keep_response2]

    response_df = response_df1.merge(response_df2, how='left', on='ANON_ID')
    response_df['RESPONSE'] = response_df.apply(lambda row: determine_response(row), axis=1)

    keep_response = response_df['RESPONSE'] != -1
    response_df = response_df[keep_response]

    label_df = pd.DataFrame(columns=['anon_id', 'text', 'label'])

    label_df['anon_id'] = df1['ANON_ID']
    # change to 'REPORT' for pathology, 'NOTE' for radiology
    label_df['text'] = df1['NOTE']

    label_df['label'] = df1.apply(lambda row: label(row['ANON_ID'], response_df), axis=1)

    keep_label = label_df['label'] != -1
    label_df = label_df[keep_label]

    print(label_df)

    label_df.to_csv(write_path, sep='|')


if __name__ == "__main__":
    label_data('../haruka_radiology_reports_111618.csv', '../V4_S_CCR_TUMOR.csv', '../surv_time.csv',
               '../time_labeled_rad_reports.csv')
