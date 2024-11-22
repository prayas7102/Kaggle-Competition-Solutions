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

from skopt import BayesSearchCV

from skopt.space import Integer, Real

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.feature_selection import SelectKBest, chi2

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import pickle

from sklearn.metrics import mean_squared_error

from imblearn.over_sampling import SMOTE


from scipy import stats

from sklearn.preprocessing import KBinsDiscretizer


from sklearn.pipeline import FunctionTransformer

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.impute import KNNImputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score

import re

from sklearn.impute import SimpleImputer

import pandas as pd

# from ydata_profiling import ProfileReport

from lightgbm import LGBMClassifier

from sklearn.model_selection import RandomizedSearchCV, cross_val_score


# In[2]:


# Load data

excel_file_path = "/kaggle/input/modified-mental-health-dataset/train.csv"

df = pd.read_csv(excel_file_path, encoding="latin-1")


# In[3]:


# # # Get unique elements for each column

# column = df.columns

# column=["Sleep Duration"]

# for x in column:

#     print("feature: ", x)

#     print("value count", df[x].value_counts())

#     print("unique values", len(df[x].unique()))

#     print("\n")


# In[ ]:





# In[ ]:


def fill_missing(df):
    df["Profession"] = df["Profession"].fillna("Student")

    df["Degree"] = df["Degree"].fillna("Unknown")

    df["Pressure"] = df["Work Pressure"].fillna(df["Academic Pressure"])

    df["Satisfaction"] = df["Job Satisfaction"].fillna(df["Study Satisfaction"])

    df["CGPA"] = df["CGPA"].fillna(0)

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    degree = {
        "BCom": "B.Com",
        "B.Com": "B.Com",
        "B.Comm": "B.Com",
        "B.Tech": "B.Tech",
        "BTech": "B.Tech",
        "B.T": "B.Tech",
        "BSc": "B.Sc",
        "B.Sc": "B.Sc",
        "Bachelor of Science": "B.Sc",
        "BArch": "B.Arch",
        "B.Arch": "B.Arch",
        "BA": "B.A",
        "B.A": "B.A",
        "BBA": "BBA",
        "BB": "BBA",
        "BCA": "BCA",
        "BE": "BE",
        "BEd": "B.Ed",
        "B.Ed": "B.Ed",
        "BPharm": "B.Pharm",
        "B.Pharm": "B.Pharm",
        "BHM": "BHM",
        "LLB": "LLB",
        "LL B": "LLB",
        "LL BA": "LLB",
        "LL.Com": "LLB",
        "LLCom": "LLB",
        "MCom": "M.Com",
        "M.Com": "M.Com",
        "M.Tech": "M.Tech",
        "MTech": "M.Tech",
        "M.T": "M.Tech",
        "MSc": "M.Sc",
        "M.Sc": "M.Sc",
        "Master of Science": "M.Sc",
        "MBA": "MBA",
        "MCA": "MCA",
        "MD": "MD",
        "ME": "ME",
        "MEd": "M.Ed",
        "M.Ed": "M.Ed",
        "MArch": "M.Arch",
        "M.Arch": "M.Arch",
        "MPharm": "M.Pharm",
        "M.Pharm": "M.Pharm",
        "MA": "MA",
        "M.A": "MA",
        "MPA": "MPA",
        "LLM": "LLM",
        "PhD": "PhD",
        "MBBS": "MBBS",
        "CA": "CA",
        "Class 12": "Class 12",
        "12th": "Class 12",
        "Class 11": "Class 11",
        "11th": "Class 11",
    }

    df["Degree"] = df["Degree"].map(degree)

    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0).astype(int)

    df["Working Professional or Student"] = (
        df["Working Professional or Student"]
        .map({"Working Professional": 1, "Student": 0})
        .fillna(0)
        .astype(int)
    )

    df["Suicide"] = (
        df["Have you ever had suicidal thoughts ?"]
        .map({"Yes": 1, "No": 0})
        .fillna(0)
        .astype(int)
    )

    ordinal_mapping = {
        "More Healthy": 1,
        "Healthy": 2,
        "Less Healthy": 3,
        "Moderate": 4,
        "Less than Healthy": 5,
        "No Healthy": 6,
        "Unhealthy": 7,
    }

    df["Dietary Habits"] = df["Dietary Habits"].apply(
        lambda x: ordinal_mapping.get(x, 0)
    )

    df["Family History of Mental Illness"] = (
        df["Family History of Mental Illness"]
        .map({"Yes": 1, "No": 0})
        .fillna(0)
        .astype(int)
    )

    sleep_duration_mapping = {
        "1-2 hours": 1,
        "2-3 hours": 2,
        "3-4 hours": 3,
        "4-5 hours": 4,
        "Less than 5 hours": 4,
        "than 5 hours": 4,
        "5-6 hours": 5,
        "6-7 hours": 6,
        "7-8 hours": 7,
        "8-9 hours": 8,
        "8 hours": 8,
        "9-11 hours": 9,
        "9-5 hours": 9,
        "9-6 hours": 9,
        "10-11 hours": 10,
        "10-6 hours": 10,
        "More than 8 hours": 11,
    }

    df["Sleep Duration"] = df["Sleep Duration"].map(sleep_duration_mapping)

    df["Stress"] = df["Pressure"] + df["Financial Stress"] - df["Satisfaction"]

    return df


def pre_process(df):
    df["Age_bin"] = KBinsDiscretizer(
        n_bins=15, encode="ordinal", strategy="quantile"
    ).fit_transform(df[["Age"]])

    df = fill_missing(df)

    df = encode(df)

    return df


df = pre_process(df)


# In[5]:


df = df.drop_duplicates()


df.to_csv("df.csv", index=False)


# def gen_eda():

#     profile = ProfileReport(

#         pd.concat([df], axis=1),

#         title="Pandas Profiling Report",

#         explorative=True,

#     )

#     profile.to_file("pandas_profiling_report.html")


# gen_eda()


# In[6]:


# Define features and target


def get_X_Y(df):
    X = df.drop(columns=["Name", "id", "Depression"])

    Y = df["Depression"]

    return X, Y


X, Y = get_X_Y(df)

# Split data into train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.10, random_state=5
)

print(X_train.shape)


# In[7]:


# Get the list of categorical column names

numerical_features = X_train.columns

categories_order = {
    "Age_bin": sorted(list(df["Age_bin"].unique())),
    "Gender": sorted(list(df["Gender"].unique())),
    "Suicide": sorted(list(df["Suicide"].unique())),
    "Pressure": sorted(list(df["Pressure"].unique())),
    "Dietary Habits": sorted(list(df["Dietary Habits"].unique())),
    "Satisfaction": sorted(list(df["Satisfaction"].unique())),
    "Family History of Mental Illness": sorted(
        list(df["Family History of Mental Illness"].unique())
    ),
    "Financial Stress": sorted(list(df["Financial Stress"].unique())),
    "Sleep Duration": sorted(list(df["Sleep Duration"].unique())),
    "Work/Study Hours": sorted(list(df["Work/Study Hours"].unique())),
    "Stress": sorted(list(df["Stress"].unique())),
}

categorical_feat_ord = list(categories_order.keys())

categorical_feat_nom = ["City", "Degree"]

numerical_features_1 = ["CGPA"]


# In[8]:


# Separate transformers for categorical and numerical features


# trf = FunctionTransformer(np.log1p, validate=True)

# trf = PowerTransformer()

# trf = FunctionTransformer(np.sqrt, validate=True)

trf = FunctionTransformer(np.sin)

# trf = StandardScaler()

# trf = MinMaxScaler()


numerical_transformer_1 = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("log", trf),
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


# In[9]:


from sklearn.model_selection import StratifiedKFold


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer_onehot, categorical_feat_nom),
        ("cat_1", categorical_transformer_ordinal, categorical_feat_ord),
        ("num", numerical_transformer_1, numerical_features_1),
    ]
)


model = LGBMClassifier(verbose=-1)


# Define the pipeline

pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])


# pipeline.fit(X_train, Y_train)


param_grid = {
    "model__n_estimators": Integer(
        300, 2000
    ),  # Increased range for more robust ensemble
    "model__max_depth": Integer(3, 12),  # Reduced upper bound to prevent overfitting
    "model__learning_rate": Real(
        0.001, 0.1, prior="log-uniform"
    ),  # Narrower range for stability
    "model__num_leaves": Integer(
        20, 100
    ),  # Reduced upper bound to control model complexity
    "model__min_child_samples": Integer(
        10, 150
    ),  # Increased minimum samples for better generalization
    "model__subsample": Real(
        0.6, 0.9, prior="uniform"
    ),  # Bagging for reduced overfitting
    "model__colsample_bytree": Real(0.6, 0.9, prior="uniform"),  # Feature subsampling
    "model__reg_alpha": Real(0.01, 50.0, prior="log-uniform"),  # L1 regularization
    "model__reg_lambda": Real(0.01, 50.0, prior="log-uniform"),  # L2 regularization
    "model__min_split_gain": Real(0.01, 0.3, prior="uniform"),  # Minimum loss reduction
    "model__feature_fraction": Real(
        0.6, 0.9, prior="uniform"
    ),  # Feature fraction for each iteration
    "model__bagging_fraction": Real(0.6, 0.9, prior="uniform"),  # New: Bagging fraction
    "model__bagging_freq": Integer(2, 10),  # New: Bagging frequency
}


# Enhanced cross-validation setup
bayes_search = BayesSearchCV(
    pipeline,
    param_grid,
    n_iter=100,  # More iterations for thorough search
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",  # More metrics
    refit="roc_auc",
    verbose=1,
    random_state=42,
    n_jobs=-1,
    n_points=3,  # More points per iteration
    return_train_score=True,
)


# Fit the model with hyperparameter tuning

bayes_search.fit(X_train, Y_train)


# Get the best parameters and model

best_pipeline = bayes_search.best_estimator_

pipeline = best_pipeline

print("Best hyperparameters:", bayes_search.best_params_)


# In[10]:


# Evaluate the tuned model

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)

print(f"Accuracy: {accuracy}")

print(
    "Cross-validation accuracy:",
    cross_val_score(pipeline, X_test, Y_test, cv=3, scoring="accuracy").mean(),
)


# In[11]:


# Save the best model

with open("best_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)


def pre_process_test(df):
    df = fill_missing(df)

    df = encode(df)

    age_bin_mapping = dict(zip(X_train["Age"], X_train["Age_bin"]))

    df["Age_bin"] = df["Age"].map(age_bin_mapping).fillna(-1)

    return df


# Generate submission as before


def generate_submission(test_file):
    df = pd.read_csv(test_file)

    df.replace("", np.nan, inplace=True)

    df = pre_process_test(df)

    filtered_df = df[X_train.columns]

    predictions = pipeline.predict(filtered_df)

    original_df = pd.read_csv(test_file)

    original_df["Target"] = predictions

    submission_df = original_df[["id", "Target"]]

    submission_df.to_csv("submission.csv", index=False)

    print("Submission file saved as 'submission.csv'")


# Run submission generation

test_file = "/kaggle/input/playground-series-s4e11/test.csv"

generate_submission(test_file)

