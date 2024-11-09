#!/usr/bin/env python
# coding: utf-8

# In[171]:


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
from ydata_profiling import ProfileReport
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score


# In[172]:


# Load data
excel_file_path = "./train.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")


# In[173]:


# # # Get unique elements for each column
# column = df.columns
# column=["Sleep Duration"]
# for x in column:
#     print("feature: ", x)
#     print("value count", df[x].value_counts())
#     print("unique values", len(df[x].unique()))
#     print("\n")


# In[174]:


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


# In[175]:


outlier_dict = {
    "normal": [],
    "skew": [],
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

def impute_fuel_type(df: pd.DataFrame)->pd.DataFrame:
     # Function to extract fuel type from engine column
    def extract_fuel_type(engine):
        match = re.search(r'(\w+)\sFuel', engine)
        return match.group(1) if match else None
    df['fuel_type'] = df.apply(lambda row: row['fuel_type'] if pd.notna(row['fuel_type']) else extract_fuel_type(row['engine']), axis=1)
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

def fill_missing(df):
    df["Profession"] = df["Profession"].fillna("Student")
    df["Degree"] = df["Degree"].fillna("Unknown")
    df['Pressure'] = df['Work Pressure'].fillna(df['Academic Pressure'])
    df['Satisfaction'] = df['Job Satisfaction'].fillna(df['Study Satisfaction'])
    df["CGPA"] = df["CGPA"].fillna(0)
    return df

def encode(df: pd.DataFrame)->pd.DataFrame:
    degree = {
        "BCom": "B.Com", "B.Com": "B.Com", "B.Comm": "B.Com",
        "B.Tech": "B.Tech", "BTech": "B.Tech", "B.T": "B.Tech",
        "BSc": "B.Sc", "B.Sc": "B.Sc", "Bachelor of Science": "B.Sc",
        "BArch": "B.Arch", "B.Arch": "B.Arch",
        "BA": "B.A", "B.A": "B.A",
        "BBA": "BBA", "BB": "BBA",
        "BCA": "BCA",
        "BE": "BE",
        "BEd": "B.Ed", "B.Ed": "B.Ed",
        "BPharm": "B.Pharm", "B.Pharm": "B.Pharm",
        "BHM": "BHM",
        "LLB": "LLB", "LL B": "LLB", "LL BA": "LLB", "LL.Com": "LLB", "LLCom": "LLB",
        "MCom": "M.Com", "M.Com": "M.Com",
        "M.Tech": "M.Tech", "MTech": "M.Tech", "M.T": "M.Tech",
        "MSc": "M.Sc", "M.Sc": "M.Sc", "Master of Science": "M.Sc",
        "MBA": "MBA",
        "MCA": "MCA",
        "MD": "MD",
        "ME": "ME",
        "MEd": "M.Ed", "M.Ed": "M.Ed",
        "MArch": "M.Arch", "M.Arch": "M.Arch",
        "MPharm": "M.Pharm", "M.Pharm": "M.Pharm",
        "MA": "MA", "M.A": "MA",
        "MPA": "MPA",
        "LLM": "LLM",
        "PhD": "PhD",
        "MBBS": "MBBS",
        "CA": "CA",
        "Class 12": "Class 12", "12th": "Class 12",
        "Class 11": "Class 11", "11th": "Class 11"
    }
    df['Degree'] = df['Degree'].map(degree)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female':0}).fillna(0).astype(int)
    df['Working Professional or Student'] = df['Working Professional or Student'].map({'Working Professional': 1, 'Student':0}).fillna(0).astype(int)
    df["Suicide"]=df["Have you ever had suicidal thoughts ?"].map({'Yes': 1, 'No':0}).fillna(0).astype(int)
    ordinal_mapping = { 'More Healthy': 1, 'Healthy': 2, 'Less Healthy': 3, 'Moderate': 4, 'Less than Healthy': 5, 'No Healthy': 6, 'Unhealthy': 7}
    df['Dietary Habits'] = df['Dietary Habits'].apply(lambda x: ordinal_mapping.get(x, 0))
    df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'Yes': 1, 'No':0}).fillna(0).astype(int)
    sleep_duration_mapping = { '1-2 hours': 1, '2-3 hours': 2, '3-4 hours': 3, '4-5 hours': 4, 'Less than 5 hours': 4, 'than 5 hours': 4, '5-6 hours': 5, '6-7 hours': 6, '7-8 hours': 7, '8-9 hours': 8, '8 hours': 8, '9-11 hours': 9, '9-5 hours': 9, '9-6 hours': 9, '10-11 hours': 10, '10-6 hours': 10, 'More than 8 hours': 11}
    df['Sleep Duration'] = df['Sleep Duration'].map(sleep_duration_mapping)
    df['Stress'] = df["Pressure"]+df["Financial Stress"]-df["Satisfaction"]
    return df

def pre_process(df):
    df['Age_bin'] = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile').fit_transform(df[['Age']])
    df = fill_missing(df)
    df = encode(df)
    return df


df = pre_process(df)
df = remove_outliers(df, outlier_dict)


# In[176]:


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


# In[177]:


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


# In[178]:


# Get the list of categorical column names
numerical_features = X_train.columns
categories_order = {
    "Age_bin": sorted(list(df["Age_bin"].unique())),
    "Gender": sorted(list(df["Gender"].unique())),
    "Suicide": sorted(list(df["Suicide"].unique())),
    "Pressure": sorted(list(df["Pressure"].unique())),
    "Dietary Habits": sorted(list(df["Dietary Habits"].unique())),
    "Satisfaction": sorted(list(df["Satisfaction"].unique())),
    "Family History of Mental Illness": sorted(list(df["Family History of Mental Illness"].unique())),
    "Financial Stress": sorted(list(df["Financial Stress"].unique())),
    "Sleep Duration": sorted(list(df["Sleep Duration"].unique())),
    "Work/Study Hours": sorted(list(df["Work/Study Hours"].unique())),
    "Stress": sorted(list(df["Stress"].unique())),
}
categorical_feat_ord = list(categories_order.keys())
categorical_feat_nom = ["City", "Degree"]
numerical_features_1 = ["CGPA"]


# In[179]:


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


# In[ ]:


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
pipeline = Pipeline([("preprocessor", preprocessor),("model", model)])

# pipeline.fit(X_train, Y_train)

from skopt.space import Integer, Real

param_grid = {
    "model__n_estimators": Integer(50, 300),
    "model__max_depth": Integer(3, 10),
    "model__learning_rate": Real(0.005, 0.2, prior='log-uniform'),
    "model__num_leaves": Integer(15, 125),
    "model__min_child_samples": Integer(5, 30),
    "model__subsample": Real(0.6, 1.0, prior='uniform'),
    "model__colsample_bytree": Real(0.6, 1.0, prior='uniform')
}

from skopt import BayesSearchCV

bayes_search = BayesSearchCV(
    pipeline,
    param_grid,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    verbose=-1,
    random_state=42,
    n_jobs=-1
)

# Fit the model with hyperparameter tuning
bayes_search.fit(X_train, Y_train)

# Get the best parameters and model
best_pipeline = bayes_search.best_estimator_
pipeline = best_pipeline
print("Best hyperparameters:", bayes_search.best_params_)


# In[181]:


# Evaluate the tuned model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Cross-validation accuracy:", cross_val_score(pipeline, X_test, Y_test, cv=3, scoring="accuracy").mean())


# In[182]:


# Save the best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

def pre_process_test(df):
    df = fill_missing(df)
    df = encode(df)
    age_bin_mapping = dict(zip(X_train['Age'], X_train['Age_bin']))
    df['Age_bin'] = df['Age'].map(age_bin_mapping).fillna(-1)
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
test_file = "test.csv"
generate_submission(test_file)

