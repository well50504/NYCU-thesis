import os
import pandas as pd
from scipy.io import savemat

def convert_csv_to_mat(directory):
    # visit all files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_Filter.csv") or file.endswith("_Filter_fix.csv"):
                csv_path = os.path.join(root, file)
                try:
                    # try to read the csv file with utf-8 encoding
                    data = pd.read_csv(csv_path, encoding='utf-8', header=None)
                except UnicodeDecodeError:
                    try:
                        data = pd.read_csv(csv_path, encoding='latin1', header=None)
                    except UnicodeDecodeError:
                        print(f"Unable to read file: {csv_path}")
                        continue

                # check if the last column is all NaN
                if data.iloc[:, -1].isna().all():
                    data = data.iloc[:, :-1]

                # remove rows
                data = data.drop(index=[15, 31])

                # csv to mat
                mat_path = os.path.splitext(csv_path)[0] + '.mat'
                # dataframe to numpy array
                data_array = data.values
                # save as mat file
                savemat(mat_path, {'data': data_array})
                print(f"Converted {csv_path} to {mat_path}")

# choose the directory to convert
main_directory = 'D:/RTMS/data/'
convert_csv_to_mat(main_directory)

