#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
from sklearn.linear_model import Lasso, LinearRegression, SGDRegressor
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
import re

import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# In[2]:


# Load data
excel_file_path = "./train.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")


# In[3]:


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
                # df[cat] = np.where(
                #     df[cat] > upper_limit,
                #     upper_limit,
                #     np.where(df[cat] < lower_limit, lower_limit, df[cat]),
                # )
                # Trimming
                df = df[(df[cat] < upper_limit) & (df[cat] > lower_limit)]
    return df


# In[4]:


outlier_dict = {
    "normal": [],
    "skew": ['milage'],
}

def reduce_engine(df: pd.DataFrame)->pd.DataFrame:
    
    def extract_capacity(x: str)->float:
        '''Extracts the volume (mentioned at 3.14L, 3.14 Litres, 3 L, 3. L)'''
        matchL = re.search( r'([.\d]+)\s*(?:L|Litres|litres|.Litres)', x)
        if bool(matchL):
            capacity = str(matchL.group(0))
            return float(re.findall(r"[-+]?\d*\.*\d+", capacity)[0])
        return np.nan
    
    def extract_horse_power(x: str)->float:        
        '''Extracts the HorsePower (mentioned at 3.14HP, 3.14 HP, 3 HP, 3. HP)'''
        matchHP = re.search( r'([.\d]+)\s*(?:HP| HP|  HP|.HP)', x)
        if bool(matchHP):
            horsePower = str(matchHP.group(0))
            return float(re.findall(r"[-+]?\d*\.*\d+", horsePower)[0])
        return np.nan
    
    def extract_cylinders(x: str)->float:        
        '''Extracts the nom of Cylinders (mentioned at 3Cylinders)'''
        matchCylinders = re.search( r'([.\d]+)\s*(?:Cylinder| Cylinder)', x)
        if bool(matchCylinders):
            cylinders = str(matchCylinders.group(0))
            return float(re.findall(r"[-+]?\d*\.*\d+", cylinders)[0])
        return 0.0

    df['engine_volume'] = df['engine'].apply(extract_capacity)
    df['engine_HP'] = df['engine'].apply(extract_horse_power)
    df['cylinders'] = df['engine'].apply(extract_cylinders)
    return df

def detect_starting_number(s: str) -> int:
        """
        Detects if string starts with positive integer and returns value otherwise returns 0
        """
        s = s.lstrip()  
        if s and s[0].isdigit():
            num = ''
            for char in s:
                if char.isdigit():
                    num += char
                else:
                    break
            return int(num) if int(num) >= 1 else 0
        return 0

def frequency_encoding(df, columns):
        for col in columns:
            freq_encoding = df[col].value_counts() / len(df)
            name = col + "_freq"
            df[name] = df[col].map(freq_encoding)
        return df

def encode_fuel(df: pd.DataFrame)->pd.DataFrame:
        df['fuel_type'] = df['fuel_type'].map({'Gasoline': 1,'Diesel': 2,'Hybrid': 3,'Plug-In Hybrid': 3,}).fillna(0).astype(int)
        return df

final_dict = {}
def encode_brand(df: pd.DataFrame)->pd.DataFrame:
    dick = dict(df.groupby('brand')['price'].mean())
    sorted_dict = dict(sorted(dick.items(), key=lambda item: item[1]))
    i=1
    for x,y in sorted_dict.items():
         final_dict[x]=i
         i+=1
    df['brand_val'] = df['brand'].map(final_dict).fillna(0).astype(int)
    return df
    
def pre_process(df):
    # df = frequency_encoding(df, ["brand", "model", "engine", "int_col", "ext_col"])
    df['model_year_bin'] = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile').fit_transform(df[['model_year']])
    df['transmission'] = df['transmission'].apply(lambda x: x.lower().replace("a/t", "automatic").replace("m/t", "manual"))
    df['speed'] = df['transmission'].apply(detect_starting_number)
    df['automatic'] = df['transmission'].apply(lambda x: 'automatic' in x).astype(int)
    df = encode_fuel(df)
    df = reduce_engine(df)
    df = encode_brand(df)
    return df


df = pre_process(df)
df = remove_outliers(df, outlier_dict)


# In[5]:


df = df.drop_duplicates()
df["price"] = np.log1p(df["price"])
df['accident'] = df['accident'].fillna("Unknown") 
df.to_csv("df.csv", index=False)


def gen_eda():
    profile = ProfileReport(
        pd.concat([df], axis=1),
        title="Pandas Profiling Report",
        explorative=True,
    )
    profile.to_file("pandas_profiling_report.html")


# gen_eda()


# In[6]:


# Define features and target
def get_X_Y(df):
    X = df.drop(columns=["id", "price"])
    Y = df["price"]
    return X, Y


X, Y = get_X_Y(df)
# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.10, random_state=5
)
print(X_train.shape)


# In[7]:


# # Get unique elements for each column
# for x in list(df.columns):
#     print("feature: ", x)
#     print("value count", df[x].value_counts())
#     print("unique values", len(df[x].unique()))
#     print("\n")


# In[8]:


# Get the list of categorical column names
numerical_features = X_train.columns
categories_order = {
    "accident": ["None reported", "At least 1 accident or damage reported"],
    "model_year": sorted(list(df["model_year"].unique())),
    "clean_title": ["No", "Yes"],
    "fuel_type": sorted(list(df["fuel_type"].unique())),
    "model_year_bin":sorted(list(df["model_year_bin"].unique())),
    "speed":sorted(list(df["speed"].unique())),
    "automatic":sorted(list(df["automatic"].unique())),
    "cylinders": sorted(list(df["cylinders"].unique())),
    "brand_val": sorted(list(df["brand_val"].unique())),
}
categorical_feat_ord = list(categories_order.keys())
categorical_feat_nom = [ "ext_col", "int_col", "brand", "model", "engine", "transmission"]
numerical_features = ["milage", "engine_volume", "engine_HP"]
# engine, transmission, ext_col, int_col, brand


# In[9]:


# Separate transformers for categorical and numerical features

from sklearn.impute import SimpleImputer


# trf = FunctionTransformer(np.log1p, validate=True)
# trf = PowerTransformer()
trf = FunctionTransformer(np.sqrt, validate=True)
# trf = FunctionTransformer(np.sin)
# Add Polynomial Features
poly=PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("log", trf),
        # ("poly", poly),
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
        ("imputer", SimpleImputer(strategy="most_frequent")),  
        (
            "ordinal",
            OrdinalEncoder(
                categories=[categories_order[col] for col in categorical_feat_ord],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
)


# In[10]:


from lightgbm import LGBMRegressor


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer_onehot, categorical_feat_nom),
        ("cat_1", categorical_transformer_ordinal, categorical_feat_ord),
        ("num", numerical_transformer, numerical_features),
    ]
)

# model = LinearRegression()
# model = Lasso(alpha=0.1)
# model = SGDRegressor(max_iter=1000, tol=1e-3, penalty='l2', random_state=42)
# model = XGBRegressor(learning_rate=0.22, n_estimators=500, subsample=1)
model = LGBMRegressor( learning_rate=0.1, n_estimators=1000, max_depth=8, num_leaves=32, min_child_samples=20, verbose=-1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=42)
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model = DecisionTreeRegressor(random_state=42)

# Define the pipeline
pipeline = Pipeline([("preprocessor", preprocessor),("model", model)])

# Fit the pipeline on the training data
pipeline.fit(X_train, Y_train)


# In[11]:


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


# In[12]:


# Define the columns expected by the model
column_names = X_train.columns

def pre_process_test(df):
    # columns = ["brand", "model", "engine", "int_col", "ext_col"]
    # # Calculate frequency encodings from X_train
    # freq_encodings = {}
    # for col in columns:
    #     freq_encodings[col] = X_train[col].value_counts() / len(X_train)
    # # Apply frequency encoding to df
    # for col in columns:
    #     if col in df.columns:
    #         df[col + '_freq'] = df[col].map(freq_encodings.get(col, {})).fillna(0)
    
    # Map 'model_year_bin' from X_train to df using 'model_year'
    # Create a mapping of model_year to model_year_bin from X_train
    model_year_bin_mapping = dict(zip(X_train['model_year'], X_train['model_year_bin']))
    # Update df's model_year_bin using this mapping
    df['model_year_bin'] = df['model_year'].map(model_year_bin_mapping).fillna(-1)
    df['transmission'] = df['transmission'].apply(lambda x: x.lower().replace("a/t", "automatic").replace("m/t", "manual"))
    df['speed'] = df['transmission'].apply(detect_starting_number)
    df['automatic'] = df['transmission'].apply(lambda x: 'automatic' in x).astype(int)
    df = encode_fuel(df)
    df = reduce_engine(df)
    df['brand_val'] = df['brand'].map(final_dict)
    return df

def generate_submission(test_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(test_file)
    df = pd.DataFrame(df)
    # Replace empty strings with NaN
    df.replace("", np.nan, inplace=True)
    # df = pre_process(df)
    df = pre_process_test(df)
    # Select the relevant columns
    filtered_df = df[column_names]
    predictions = pipeline.predict(filtered_df)
    # Load the original test file to keep the PassengerId column
    original_df = pd.read_csv(test_file)
    original_df["price"] = predictions
    original_df["price"] = np.expm1(original_df["price"])
    # Save the results to a new CSV file
    submission_df = original_df[["id", "price"]]
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved as 'submission.csv'")


# Generate the submission
test_file = "test.csv"
generate_submission(test_file)

