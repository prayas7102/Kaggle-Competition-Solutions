#!/usr/bin/env python
# coding: utf-8

# In[517]:


'''run ridge,lasso,tree regressors'''
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
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE

from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# model = LinearRegression()
# model = Lasso(alpha=0.1)
model = XGBRegressor(objective='reg:squarederror', scale_pos_weight=1)  
# model = DecisionTreeRegressor(random_state=42)


# In[518]:


# Load data
excel_file_path = "./train.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")


# In[519]:


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


# In[520]:


# how to know no. of bins
outlier_dict = {
    "normal": [
    ],
    "skew": [
        # "milage"
    ],
}


def pre_process(df):
    
    return df


df = pre_process(df)
df = remove_outliers(df, outlier_dict)


# In[521]:


df.head()


# In[522]:


df.isnull().sum()


# In[523]:


df = df.drop_duplicates()
df['price'] = np.log1p(df['price'])

df.to_csv("df.csv", index=False)

def gen_eda():
    profile = ProfileReport(
        pd.concat([df], axis=1),
        title="Pandas Profiling Report",
        explorative=True,
    )
    profile.to_file("pandas_profiling_report.html")


# gen_eda()


# In[524]:


# Define features and target
def get_X_Y(df):
    X = df.drop(columns=["id", "price", "clean_title"])
    Y = df["price"]
    return X, Y

X, Y = get_X_Y(df)
# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=5
)
print(X_train.shape)


# In[525]:


# Get unique elements for each column
for x in list(df.columns):
    print('feature: ', x)
    print('value count', df[x].value_counts())
    print('unique values',len(df[x].unique()))
    print('\n')


# In[526]:


# Get the list of categorical column names
numerical_features = X_train.columns
# ordinal data
# Define the categories in the order you want
year = sorted(list(df['model_year'].unique()))
title = ['No', 'Yes']
categories_order = {
    "accident": ['None reported', 'At least 1 accident or damage reported'],
    "model_year": year,
}
categorical_feat_ord = list(categories_order.keys())
categorical_feat_nom = [
    "ext_col", "int_col", "brand", "model", "fuel_type", "engine", "transmission"
]
cat = categorical_feat_ord + categorical_feat_nom
numerical_features = [item for item in numerical_features if item not in cat]


# In[527]:


# Separate transformers for categorical and numerical features

from sklearn.impute import SimpleImputer


trf = FunctionTransformer(np.log1p, validate=True)
# Add Polynomial Features
# poly = PolynomialFeatures(degree=2, include_bias=False)

numerical_transformer = Pipeline(
    steps=[
        # ("poly", poly),
        ("log", trf)
    ]
)
categorical_transformer_onehot = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# Create the categorical transformer for ordinal features with an imputer
categorical_transformer_ordinal = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values with the most frequent value
        ("ordinal", OrdinalEncoder(categories=[categories_order[col] for col in categorical_feat_ord],
                                   handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
)


# In[528]:


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


# In[529]:


# # Calculate the correlation matrix
# correlation_matrix = df.corr()

# # Save the correlation matrix to a CSV file
# correlation_matrix.to_csv('correlation_matrix.csv', index=True)


# In[530]:


# Save the fitted pipeline as a .pkl file
filename_pkl = "model.pkl"
pickle.dump(pipeline, open(filename_pkl, "wb"))
print(f"Model saved as {filename_pkl}")
# Evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(Y_test, y_pred)
n = len(Y_test)
p = 1
adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
print(f"Adjusted R² score: {adj_r2}")


# In[531]:


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
    original_df["price"] = predictions
    original_df['price'] = np.expm1(original_df['price'])
    # Save the results to a new CSV file
    submission_df = original_df[["id", "price"]]
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved as 'submission.csv'")


# Generate the submission
test_file = "test.csv"
generate_submission(test_file)
