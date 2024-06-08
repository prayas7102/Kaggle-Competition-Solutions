# Import necessary libraries
import os
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")

# Load data
excel_file_path = "./train.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")
df = pd.DataFrame(df)


def extract_first_last(df):
    return df


df = extract_first_last(df)
df.columns
df = df.drop_duplicates()
df = df.drop(columns=["id", "Target"])
# Assuming df is your DataFrame

cat_feat = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
    "Educational special needs", "International"
]


def draw_graph(df, save_dir="plots"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for col in df.columns:
        plt.figure(figsize=(14, 4))
        # First subplot: Histogram with KDE
        plt.subplot(121)
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} - Histogram")
        # Second subplot: QQ plot
        plt.subplot(122)
        stats.probplot(df[col], dist="norm", plot=plt)
        plt.title(f"{col} - QQ Plot")
        # Save the figure
        col = col.replace("/", " ")
        plt.savefig(os.path.join(save_dir, f"{col}_plot.png"))
        # Close the figure to free up memory
        plt.close()


# before transformation
draw_graph(df)
