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
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")

# Load data
excel_file_path = "./df.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")
df = pd.DataFrame(df)


def extract_first_last(df):
    return df


df = extract_first_last(df)
df.columns
df = df.drop_duplicates()
df = df.drop(columns=["id", "Target", "Educational special needs", "International"])
# Assuming df is your DataFrame


def draw_graph(df, save_dir="plots-1"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    colu = df.columns
    colu = ["GDP"]
    for col in colu:
        plt.figure(figsize=(14, 4))
        # First subplot: Histogram with KDE
        plt.subplot(121)
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} - Histogram {df[col].skew()}")
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
# draw_graph(df)

# corr_matrix = df.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
# plt.savefig('correlation_matrix_heatmap.png')

# num-num

# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=df, x='Curricular units 1st sem (enrolled)', y='Curricular units 2nd sem (enrolled)')
# plt.title('Scatter Plot of Variable1 vs Variable2')
# plt.show()

# num-cat
df = pd.read_csv(excel_file_path, encoding="latin-1")
sns.barplot(x=df["Target"], y=df["Course"], hue=df["Target"])
plt.show()