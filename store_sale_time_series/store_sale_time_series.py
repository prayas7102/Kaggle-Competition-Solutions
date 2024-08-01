#!/usr/bin/env python
# coding: utf-8

# In[31]:


'''combining col, including other csv, corr'''
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
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
from pandas_profiling import ProfileReport

warnings.filterwarnings("ignore")

model = LinearRegression()


# In[32]:


# Load data
excel_file_path = "./train.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")
stores_df = pd.read_csv('stores.csv')
# oil
oil_df = pd.read_csv('oil.csv')
imp_oil = int(oil_df['dcoilwtico'].mean())
imp_oil = 0
oil_df['dcoilwtico'] = oil_df['dcoilwtico'].fillna(imp_oil)
# transactions
transactions_df = pd.read_csv('transactions.csv')
imp_transaction = int(transactions_df['transactions'].mean())
imp_transaction = 0
transactions_df['transactions'] = transactions_df['transactions'].fillna(imp_transaction)


# In[33]:


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


# In[34]:


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
    df['date_dow'] = df['date'].dt.dayofweek
    df['date_is_weekend'] = np.where(df['date_dow'].isin([5,6]), 1,0)
    # Convert the specified columns to object type
    columns_to_convert = ['month', 'year', 'date_dow', 'date_is_weekend']
    df[columns_to_convert] = df[columns_to_convert].astype('object')
    # store nbr
    df = pd.merge(df, stores_df, on='store_nbr', how='left')
    # dcoilwtico
    oil_df['date'] = pd.to_datetime(oil_df['date'])
    df = pd.merge(df, oil_df, on='date', how='left')
    df['dcoilwtico'] = df['dcoilwtico'].fillna(int(imp_oil))
    # transaction
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    df = pd.merge(df, transactions_df, on=['date', 'store_nbr'], how='left')
    df['transactions'] = df['transactions'].fillna(int(imp_transaction))
    return df


df = pre_process(df)
df = remove_outliers(df, outlier_dict)


# In[35]:


df.head()


# In[36]:


df = df.drop_duplicates()
df['sales'] = np.log1p(df['sales'])
df.to_csv("df.csv", index=False)

def gen_eda():
    profile = ProfileReport(
        pd.concat([df], axis=1),
        title="Pandas Profiling Report",
        explorative=True,
    )
    profile.to_file("pandas_profiling_report.html")


# gen_eda()


# In[37]:


# Define features and target
def get_X_Y(df):
    X = df.drop(columns=["id","sales","date", "dcoilwtico", "transactions"])
    Y = df["sales"]
    return X, Y

X, Y = get_X_Y(df)


# In[38]:


# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=5
)
print(X_train.shape)


# In[39]:


# Get the list of categorical column names
numerical_features = X_train.columns
categorical_feat_ord = [
    # "dcoilwtico", "transactions"
]
categorical_feat_nom = [
    'store_nbr', 'family', 
    'month', 'year', 'date_dow', 
    'city', 'state', 'type', 'cluster'
]
cat = categorical_feat_ord + categorical_feat_nom
numerical_features = [item for item in numerical_features if item not in cat]
numerical_features = ['onpromotion']


# In[40]:


# Separate transformers for categorical and numerical features

trf = FunctionTransformer(np.log1p, validate=True)
# Add Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)

numerical_transformer = Pipeline(
    steps=[
        ("poly", poly),
        ("log", trf)
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


# In[41]:


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


# In[42]:


# # Calculate the correlation matrix
# correlation_matrix = df.corr()

# # Save the correlation matrix to a CSV file
# correlation_matrix.to_csv('correlation_matrix.csv', index=True)


# In[43]:


# Save the fitted pipeline as a .pkl file
filename_pkl = "model.pkl"
pickle.dump(pipeline, open(filename_pkl, "wb"))
print(f"Model saved as {filename_pkl}")


# In[44]:


# Evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[45]:


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
    original_df['sales'] = np.expm1(original_df['sales'])
    # Save the results to a new CSV file
    submission_df = original_df[["id", "sales"]]
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved as 'submission.csv'")


# Generate the submission
test_file = "test.csv"
generate_submission(test_file)

