# Import necessary libraries
import os
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import dask.dataframe as dd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

warnings.filterwarnings("ignore")

# Load data
excel_file_path = "./df.csv"
df = dd.read_csv(excel_file_path)
df = df.compute()

def draw_graph(df, colu, save_dir="plots-1"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df["CGPA"] = FunctionTransformer(np.sin).fit_transform(df[["CGPA"]])
    df["CGPA"] = PowerTransformer().fit_transform(df[["CGPA"]])
    for col in colu:
        # print(df[col])
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

# draw_graph(df, ["CGPA"])

# categorical data analysiss
# Create a directory if it doesn't exist
output_dir = './plots-1'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# Define a function to plot and save histogram of a specific feature
def plot_histogram(df, feature, bins=30, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], bins=bins, kde=True, color='blue')
    plt.title(f'Distribution of {feature}', fontsize=15)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Save the plot
    if save_path:
        save_path = os.path.join(output_dir, save_path)
        plt.savefig(save_path)
        print(f'Histogram of {feature} saved at {save_path}')
    plt.show()

plot_histogram(df, 'Stress', bins=30, save_path='Stress')
