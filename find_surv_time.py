import pandas as pd


def surv_time(row):
    year1 = int(row['DATEDX'][-2:])
    if year1 < 20:
        year1 = year1 + 100
    year2 = int(row['FUDATE'][-2:])
    if year2 < 20:
        year2 = year2 + 100
    return year2 - year1


def label_data(read_path_1, read_path_2, write_path):
    df1 = pd.read_csv(read_path_1, sep=',')
    df2 = pd.read_csv(read_path_2, sep=',')

    df1 = df1.merge(df2, on='ANON_ID')

    df1['SURV_TIME'] = df1.apply(lambda row: surv_time(row), axis=1)

    print(df1)

    df1.to_csv(write_path, sep='|')


if __name__ == "__main__":
    label_data('../V4_S_CCR_TUMOR.csv', '../V4_S_CCR_COMORBID.csv', '../surv_time.csv')
