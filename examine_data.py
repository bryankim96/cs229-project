import re

import pandas as pd


def find_in_reports(path, search_strs, find='all', text_col_name='REPORT', max_find=-1, window_size=10):
    df = pd.read_csv(path, sep='|')

    i = 0
    for index, row in df.iterrows():
        report = row[text_col_name]
        if 0 <= max_find <= i:
            break
        output = ""
        find_results = []
        for search_str in search_strs:
            result = re.search(search_str, report)
            if result is None:
                find_results.append(-1)
            else:
                find_results.append(result.start())
        if find == 'all' and all(i >= 0 for i in find_results):
            for pos in find_results:
                start = max(pos - window_size, 0)
                end = min(pos + window_size, len(report))
                output += (report[start:end] + '\t')
            i += 1
            print(index, row['ANON_ID'], output, "\n")
        elif find == 'any' and any(i >= 0 for i in find_results):
            for pos in find_results:
                if pos >= 0:
                    start = max(pos - window_size, 0)
                    end = min(pos + window_size, len(report))
                    output += (report[start:end] + '\t')
            i += 1
            print(index, row['ANON_ID'], output, "\n")

    print("Num matching reports found: {}/{}".format(i, len(df)))


def get_class_counts(path, label_values=None, label_col_name='label'):
    # change sep to '|' or ',' depending on file
    df = pd.read_csv(path, sep='|')
    print(df[label_col_name].unique())

    if label_values:
        counts = {label: 0 for label in label_values}
    else:
        label_values = df[label_col_name].unique()
        counts = {label: 0 for label in label_values}

    for label in df[label_col_name]:
        if label in counts:
            counts[label] += 1

    for label in counts:
        print("Class {}: {}/{} ({}%)".format(label, counts[label], len(df), 100.0 * (counts[label] / len(df))))


if __name__ == "__main__":
    # find_in_reports('../haruka_pathology_reports_111618.csv', [' yp', '(?<![C])T0', 'N0'])
    # find_in_reports('../haruka_pathology_reports_111618.csv', [' yp'])
    # find_in_reports('../haruka_radiology_reports_111618.csv', ['T0', 'N0'], text_col_name='NOTE')

    get_class_counts('../new_labeled_reports_full_preprocessed.csv', label_col_name='label')
