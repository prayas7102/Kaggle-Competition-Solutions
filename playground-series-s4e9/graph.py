# Import necessary libraries
import os
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import dask.dataframe as dd

warnings.filterwarnings("ignore")

# Load data
excel_file_path = "./df.csv"
df = dd.read_csv(excel_file_path)
df = df.compute()

df = df.drop_duplicates()
# Assuming df is your DataFrame

def draw_graph(df, save_dir="plots-1"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    colu = df.columns
    colu = ["milage"]
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

# draw_graph(df)

# Create a directory if it doesn't exist
output_dir = './plots-1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define a function to plot and save histogram of a specific feature
def plot_histogram(df, feature, bins=30, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], bins=bins, kde=False, color='blue')
    plt.title(f'Distribution of {feature}', fontsize=15)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Save the plot
    if save_path:
        save_path = os.path.join(output_dir, save_path)
        plt.savefig(save_path)
        print(f'Histogram of {feature} saved at {save_path}')
    plt.show()

# Example: plot and save histogram for 'milage' feature
plot_histogram(df, 'model_year', bins=30, save_path='model_year.png')

plot_histogram(df, 'model_year_bin', bins=30, save_path='model_year_bin.png')
