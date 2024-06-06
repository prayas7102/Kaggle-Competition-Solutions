# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load data
excel_file_path = "./train.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")
df = pd.DataFrame(df)


def extract_first_last(df):
    df[["deck", "num", "side"]] = df["Cabin"].str.split("/", expand=True)
    df["group"] = df["PassengerId"].str[:4]
    df["family_size"] = [list(df["group"]).count(x) for x in list(df["group"])]
    df["Age_Cat"] = pd.cut(
        df["Age"],
        bins=[0, 18, 30, 50, 80],
        labels=["Child", "Young Adult", "Adult", "Senior"],
    )
    df["room X Age"] = df["RoomService"] * df["Age"]
    return df


df = extract_first_last(df)
df.columns
df = df.drop_duplicates()
df.replace("", np.nan, inplace=True)
df.head()

plt.figure(figsize=(14, 6))
cat_feat = ["Age_Cat"]
num_feat = ["RoomService"]
# Assuming df is your DataFrame
combined_df = df.copy()
combined_df = combined_df[:500]

# Fill categorical features with their mode
for cat in cat_feat:
    mode_value = combined_df[cat].mode()[0]
    combined_df[cat] = combined_df[cat].fillna(mode_value)

# Fill numerical features with their mode
for num in num_feat:
    mode_value = combined_df[num].mode()[0]
    combined_df[num] = combined_df[num].fillna(mode_value)

# Plot
plt.figure(figsize=(14, 6))

# Box plot
plt.subplot(1, 2, 1)
sns.boxplot(
    x=cat_feat[0], y=num_feat[0], data=combined_df, palette="Set3", hue="Transported"
)
plt.title(f"Box plot of {num_feat[0]} by {cat_feat[0]}")

# Swarm plot
plt.subplot(1, 2, 2)
sns.swarmplot(
    x=cat_feat[0],
    y=num_feat[0],
    data=combined_df,
    palette="Set3",
    alpha=0.5,
    hue="Transported",
)
plt.title(f"Swarm plot of {num_feat[0]} by {cat_feat[0]}")

plt.tight_layout()
plt.show()
