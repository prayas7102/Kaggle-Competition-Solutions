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

# plot_histogram(df, 'Stress', bins=30, save_path='Stress')

def cat_graph(df, feat):
    # Group by city and count suicides
    city_suicide_count = df[df['Suicide'] == 1].groupby(feat)['Suicide'].count()
    # Plot the data
    city_suicide_count.sort_values(ascending=False).plot(kind='bar', color='skyblue', figsize=(10, 6))
    # Customize the plot
    plt.title('Suicide Counts by City')
    plt.xlabel(feat)
    plt.ylabel('Number of Suicides')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(output_dir, feat)
    plt.savefig(save_path)
    # Show the plot

# cat_graph(df, 'City')
# cat_graph(df, 'Profession')
# cat_graph(df, 'Degree')
cat_graph(df, 'Dietary Habits')

def num_graph(df, feat):
    # Plotting
    sns.histplot(data=df, x=feat, hue='Suicide', kde=True, stat="density", common_norm=False, bins=10, palette='Set2')
    plt.title('CGPA vs. Suicide Prone Distribution')
    plt.xlabel('CGPA')
    plt.ylabel('Density')
    plt.show()

# num_graph(df, 'CGPA')