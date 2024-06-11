# %%
# Import necessary libraries
# seelctive trf on sin data
# outlier removal after power trf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.linear_model import LogisticRegression
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

# %%
# highest accuracy model
# model = LGBMClassifier(verbose=-1)
# model = HistGradientBoostingClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = AdaBoostClassifier()
# model = MLPClassifier()

voting_clf = VotingClassifier(
    estimators=[
        ("ab", RandomForestClassifier()),
        ("gb", GradientBoostingClassifier()),
        ("lgbm", LGBMClassifier(verbose=-1)),
    ],
    voting="hard",  # 'hard' for majority voting, 'soft' for weighted average probabilities
)
# RandomForestClassifier(class_weight='balanced', n_estimators=100)
# model = LGBMClassifier(verbose=-1)
model = voting_clf

# %%
# Load data
excel_file_path = "./train.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")
# df.columns

# %%
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

# %%
from scipy import stats
from sklearn.mixture import GaussianMixture


df = df.drop_duplicates()
outlier_dict = {
    "normal": [
        "Admission grade",
        "Previous qualification (grade)",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ],
    "skew": [
        # "Curricular units 1st sem (evaluations)",
        # "Curricular units 2nd sem (evaluations)",
        # "Curricular units 1st sem (approved)",
        # "Curricular units 2nd sem (approved)",
        # "Age at enrollment"
    ],
}

def pre_process(df):
    # Binning 'Age at enrollment'
    bins = [17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    labels = ["17-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70"]
    df['Age at enrollment'] = df['Age at enrollment'].clip(lower=bins[0], upper=bins[-1])
    df['Age at enrollment'] = pd.cut(df['Age at enrollment'], bins=bins, labels=labels, right=True, include_lowest=True)
    return df


df = pre_process(df)
df = remove_outliers(df, outlier_dict)

# %%
from imblearn.over_sampling import SMOTE


# Define features and target
def get_X_Y(df):
    X = df.drop(
        columns=["id", "Target", "Educational special needs", "International"]
    )  # "Educational special needs", "International"
    Y = df["Target"]
    # X, Y = SMOTE(random_state=42).fit_resample(X, Y)
    return X, Y


X, Y = get_X_Y(df)

# %%
# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=5
)
# Check columns
X_train, X_test = X, X
Y_train, Y_test = Y, Y
print(X_train.shape)

# %%
# Get the list of categorical column names
numerical_features = X_train.columns
categorical_feat_ord = [
    "Scholarship holder",
    "Daytime/evening attendance",
    "Previous qualification",
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
]
categorical_feat_nom = [
    "Gender",
    "Age at enrollment",
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Nacionality",
]
cat = categorical_feat_ord+categorical_feat_nom
numerical_features = [item for item in numerical_features if item not in cat]

# %%
# import pandas as pd
# from pandas_profiling import ProfileReport


# def gen_eda():
#     profile = ProfileReport(
#         pd.concat([X_train, Y_train], axis=1),
#         title="Pandas Profiling Report",
#         explorative=True,
#     )
#     profile.to_file("pandas_profiling_report.html")


# gen_eda()

# %%
# Separate transformers for categorical and numerical features

from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

trf = PowerTransformer()
# trf = FunctionTransformer(np.sin)

numerical_transformer = Pipeline(
    steps=[
        ("log", trf),
        ("scaler", StandardScaler()),  # StandardScaler MinMaxScaler
    ]
)
categorical_transformer_onehot = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore")),]
)
categorical_transformer_ordinal = Pipeline(
    steps=[("ord", OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)),]
)

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer_onehot, categorical_feat_nom),
        ("cat_1", categorical_transformer_ordinal, categorical_feat_ord),
        ("num", numerical_transformer, numerical_features)
    ]
)
# Define the pipeline
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
# Fit the pipeline on the training data
pipeline.fit(X_train, Y_train)

# %%
# Save the fitted pipeline as a .pkl file
filename_pkl = "model.pkl"
pickle.dump(pipeline, open(filename_pkl, "wb"))
print(f"Model saved as {filename_pkl}")

# %%
# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy}")

# %%
print(classification_report(Y_test, y_pred))

# %%
cross_val_score(pipeline, X_test, Y_test, cv=3, scoring="accuracy").mean()

# %%
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
    df = pre_process(df)
    # Select the relevant columns
    filtered_df = df[column_names]
    predictions = pipeline.predict(filtered_df)
    # Load the original test file to keep the PassengerId column
    original_df = pd.read_csv(test_file)
    original_df["Target"] = predictions
    # Save the results to a new CSV file
    submission_df = original_df[["id", "Target"]]
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved as 'submission.csv'")


# Generate the submission
test_file = "test.csv"
generate_submission(test_file)