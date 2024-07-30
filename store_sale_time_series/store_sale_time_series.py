#!/usr/bin/env python
# coding: utf-8

# In[560]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE

from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import pandas as pd
from pandas_profiling import ProfileReport

warnings.filterwarnings("ignore")

model = LinearRegression()


# In[561]:


# Load data
excel_file_path = "./train.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")


# In[562]:


def remove_outliers(df, outlier_dict):
    for distribution, category in outlier_dict.items():
        if distribution == "normal":
            for cat in category:
                upper_limit = df[cat].mean() + 3 * df[cat].std()
                lower_limit = df[cat].mean() - 3 * df[cat].std()
                print(cat, upper_limit, lower_limit)
                # capping
                # df[cat] = np.where(df[cat] > upper_limit,upper_limit,np.where(df[cat] < lower_limit, lower_limit, df[cat]))
                # Trimming
                df = df[(df[cat] < upper_limit) & (df[cat] > lower_limit)]
        elif distribution == "skew":
            for cat in category:
                percentile25 = df[cat].quantile(0.25)
                percentile75 = df[cat].quantile(0.75)
                iqr = percentile75 - percentile25
                upper_limit = percentile75 + 1.5 * iqr
                lower_limit = percentile25 - 1.5 * iqr
                print(cat, upper_limit, lower_limit)
                # capping
                df[cat] = np.where(
                    df[cat] > upper_limit,
                    upper_limit,
                    np.where(df[cat] < lower_limit, lower_limit, df[cat]),
                )
                # Trimming
                # df = df[(df[cat] < upper_limit) & (df[cat] > lower_limit)]
    return df


# In[563]:


# how to know no. of bins
outlier_dict = {
    "normal": [
    ],
    "skew": [
    ],
}


def pre_process(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    # df['date_day'] = df['date'].dt.day
    df['date_dow'] = df['date'].dt.dayofweek
    df['date_is_weekend'] = np.where(df['date_dow'].isin([5,6]), 1,0)
    # Convert the specified columns to object type
    columns_to_convert = ['month', 'year', 'date_dow', 'date_is_weekend']
    df[columns_to_convert] = df[columns_to_convert].astype('object')
    df=df.drop(columns=["date"])
    # df['Age at enrollment'] = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='quantile').fit_transform(df[['Age at enrollment']])
    # df['Application mode'] = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='kmeans').fit_transform(df[['Application mode']])
    return df


df = pre_process(df)
df = remove_outliers(df, outlier_dict)


# In[564]:


df = df.drop_duplicates()
df.to_csv("df.csv", index=False)

def gen_eda():
    profile = ProfileReport(
        pd.concat([df], axis=1),
        title="Pandas Profiling Report",
        explorative=True,
    )
    profile.to_file("pandas_profiling_report.html")


# gen_eda()


# In[565]:


# Define features and target
def get_X_Y(df):
    X = df.drop(
        columns=[
            "id",
            "sales",
        ]
    )
    Y = df["sales"]
    return X, Y


X, Y = get_X_Y(df)


# In[566]:


# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=5
)

print(X_train.shape)


# In[567]:


# Calculate the correlation matrix
# correlation_matrix = df.corr()

# # Save the correlation matrix to a CSV file
# correlation_matrix.to_csv('correlation_matrix.csv', index=True)


# In[568]:


# Get the list of categorical column names
numerical_features = X_train.columns
categorical_feat_ord = [
]
categorical_feat_nom = [
    'store_nbr', 'family','month', 'year', 'date_dow'
]
cat = categorical_feat_ord + categorical_feat_nom
numerical_features = [item for item in numerical_features if item not in cat]
numerical_features = ['onpromotion']
print(numerical_features)


# In[569]:


X_train.info()


# In[570]:


Y_train.info()


# In[571]:


# Separate transformers for categorical and numerical features

trf = PowerTransformer()
trf1 = FunctionTransformer(np.log1p, validate=True)

numerical_transformer = Pipeline(
    steps=[
        ("log", trf1),
        # ("power", trf),
        # ('pca',PCA(n_components=1)) 
    ]
)
categorical_transformer_onehot = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
categorical_transformer_ordinal = Pipeline(
    steps=[
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
)


# In[572]:


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer_onehot, categorical_feat_nom),
        ("cat_1", categorical_transformer_ordinal, categorical_feat_ord),
        ("num", numerical_transformer, numerical_features),
    ]
)
# Define the pipeline
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

# Fit the pipeline on the training data
pipeline.fit(X_train, Y_train)


# In[573]:


# Save the fitted pipeline as a .pkl file
filename_pkl = "model.pkl"
pickle.dump(pipeline, open(filename_pkl, "wb"))
print(f"Model saved as {filename_pkl}")


# In[574]:


# Evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[575]:


# Define the columns expected by the model
column_names = X_train.columns


def generate_submission(test_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(test_file)
    df = pd.DataFrame(df)
    # Replace empty strings with NaN
    df.replace("", np.nan, inplace=True)
    df = pre_process(df)
    # Select the relevant columns
    filtered_df = df[column_names]
    predictions = pipeline.predict(filtered_df)
    # Load the original test file to keep the PassengerId column
    original_df = pd.read_csv(test_file)
    original_df["sales"] = predictions
    # Save the results to a new CSV file
    submission_df = original_df[["id", "sales"]]
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved as 'submission.csv'")


# Generate the submission
test_file = "test.csv"
generate_submission(test_file)

