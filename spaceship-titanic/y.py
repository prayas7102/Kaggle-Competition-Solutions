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
    AdaBoostClassifier,
    StackingClassifier,
    VotingClassifier,
)
from xgboost import XGBClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.neural_network import MLPClassifier

# from hmmlearn import hmm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle

warnings.filterwarnings("ignore")
# highest accuracy model
# model = LGBMClassifier(verbose=-1)
# model = HistGradientBoostingClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = AdaBoostClassifier()
# model = MLPClassifier()
voting_clf = VotingClassifier(
    estimators=[
        ("ab", AdaBoostClassifier()),
        ("gb", GradientBoostingClassifier()),
        ("lgbm", LGBMClassifier(verbose=-1)),
    ],
    voting="hard",  # 'hard' for majority voting, 'soft' for weighted average probabilities
)
from sklearn.model_selection import GridSearchCV


model = GradientBoostingClassifier()

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.8, 0.9, 1.0],
}

# Initialize the model
model = GradientBoostingClassifier(random_state=42, warm_start=True)

# Initialize GridSearchCV
model = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)

# Load data
excel_file_path = "./train.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")


def extract_first_last(df):
    df[["deck", "num", "side"]] = df["Cabin"].str.split("/", expand=True)
    df["group"] = df["PassengerId"].str[:4]
    df["family_size"] = [list(df["group"]).count(x) for x in list(df["group"])]
    df["Age_Cat"] = pd.cut(
        df["Age"],
        bins=[0, 18, 30, 50, 80],
        labels=["Child", "Young Adult", "Adult", "Senior"],
    )
    return df


df = extract_first_last(df)
df.columns
df = df.drop_duplicates()
df.head()

# Columns to check
columns_to_check = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
# Filter rows where all specified columns are zero
rows_to_remove = df[df[columns_to_check].eq(0).all(axis=1)]
rows_to_remove = pd.DataFrame(rows_to_remove.iloc[:2000])
print(len(df), len(rows_to_remove))
df = df[~df.index.isin(rows_to_remove.index)]
print(len(df))
df.describe()


# Define features and target
def get_X_Y(df):
    X = df.drop(
        columns=["PassengerId", "Name", "Transported", "Cabin", "group", "num"]
    )  # , "num", "side", "family_size"
    Y = df["Transported"]
    return X, Y


X, Y = get_X_Y(df)
# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=5
)
# Check columns
# X_train, X_test = X,X
# Y_train, Y_test = Y,Y
print(X_train.columns, X_train.shape)

# Get the list of numerical column names
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Get the list of categorical column names
categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()
# separte one hot and ordinal
categorical_features_ordinal = ["VIP", "Age_Cat"]
categorical_features_onehot = list(
    set(categorical_features) - set(categorical_features_ordinal)
)
print(categorical_features_ordinal, categorical_features_onehot)

X_train.isnull().sum()

# Get unique elements for each column
for x in categorical_features:
    print(x, X_train[x].unique(), len(X_train[x].unique()))


# import pandas as pd
# from pandas_profiling import ProfileReport
# def gen_eda():
#     profile = ProfileReport(pd.concat([X_train, Y_train], axis=1), title='Pandas Profiling Report', explorative=True)
#     profile.to_file("pandas_profiling_report.html")
# gen_eda()
# Separate transformers for categorical and numerical features

from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

trf = PowerTransformer()

# def square(x):
#     return x ** 2
# trf = FunctionTransformer(func=square, validate=True)


categorical_transformer_onehot = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
categorical_transformer_ordinal = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OrdinalEncoder()),
    ]
)
numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            KNNImputer(n_neighbors=5),
        ),  # KNNImputer(n_neighbors=5) SimpleImputer(strategy='mean')
        ("log", trf),
        ("scaler", StandardScaler()),  # StandardScaler MinMaxScaler
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer_onehot, categorical_features_onehot),
        ("cat_1", categorical_transformer_ordinal, categorical_features_ordinal),
        ("num", numerical_transformer, numerical_features),
    ]
)

# Define the pipeline
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
# Fit the pipeline on the training data
pipeline.fit(X_train, Y_train)
Pipeline(
    steps=[
        (
            "preprocessor",
            ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        ["side", "Destination", "CryoSleep", "deck", "HomePlanet"],
                    ),
                    (
                        "cat_1",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OrdinalEncoder()),
                            ]
                        ),
                        ["VIP", "Age_Cat"],
                    ),
                    (
                        "num",
                        Pipeline(
                            steps=[
                                ("imputer", KNNImputer()),
                                ("log", PowerTransformer()),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        [
                            "Age",
                            "RoomService",
                            "FoodCourt",
                            "ShoppingMall",
                            "Spa",
                            "VRDeck",
                            "family_size",
                        ],
                    ),
                ]
            ),
        ),
        ("model", GradientBoostingClassifier()),
    ]
)

# # Combine X_train and Y_train into a single DataFrame
# X_train_processed = pipeline.named_steps['preprocessor'].transform(X_train)
# combined_df = pd.DataFrame(X_train_processed.copy())  # Create a copy of X_train
# combined_df['Transported'] = list(Y_train.copy())  # Add the target column
# Save the fitted pipeline as a .pkl file
filename_pkl = "model.pkl"
pickle.dump(pipeline, open(filename_pkl, "wb"))
print(f"Model saved as {filename_pkl}")

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy}")
Accuracy: 0.806572068707991
print(classification_report(Y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(Y_test, y_pred)}")


cross_val_score(pipeline, X_test, Y_test, cv=3, scoring="accuracy").mean()

import pandas as pd
import numpy as np
import pickle

# Load the trained model
loaded_model = pickle.load(open("model.pkl", "rb"))

# Define the columns expected by the model
column_names = X_train.columns


def generate_submission(test_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(test_file)
    df = pd.DataFrame(df)
    # Replace empty strings with NaN
    df.replace("", np.nan, inplace=True)
    df = extract_first_last(df)
    # Select the relevant columns
    filtered_df = df[column_names]
    predictions = pipeline.predict(filtered_df)
    # Load the original test file to keep the PassengerId column
    original_df = pd.read_csv(test_file)
    original_df["Transported"] = predictions
    # Save the results to a new CSV file
    submission_df = original_df[["PassengerId", "Transported"]]
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved as 'submission.csv'")


# Generate the submission
test_file = "test.csv"
generate_submission(test_file)
