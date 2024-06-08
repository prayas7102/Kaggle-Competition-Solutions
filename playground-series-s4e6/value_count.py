# Load data
import pandas as pd
from sklearn.model_selection import train_test_split


excel_file_path = "./train.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")


def extract_first_last(df):
    return df


df = extract_first_last(df)
df.columns
df = df.drop_duplicates()
df.head()

for x in df.columns:
    print(df[x].value_counts())
