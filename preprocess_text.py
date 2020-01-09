import csv
import os

from wordsegment import load

from generate_embeddings import apply_preprocessing

if __name__ == "__main__":
    in_file_paths = ["../new_labeled_path_reports.csv", "../new_labeled_rad_reports.csv"]

    load()

    for file_path in in_file_paths:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')
            out_file_path = os.path.splitext(file_path)[0] + "_preprocessed.csv"
            out_file = open(out_file_path, "w")
            writer = csv.writer(out_file, delimiter="|") 
            writer.writerow(["anon_id", "text", "label"]) 
            header = next(reader)
            i = 0
            for row in reader:
                row[1] = " ".join(apply_preprocessing(row[1]))
                if row[2] == "-1":
                    row[2] = "0" 
                writer.writerow(row)
