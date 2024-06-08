# Load data
import pandas as pd
from sklearn.model_selection import train_test_split


excel_file_path = "./train.csv"
df = pd.read_csv(excel_file_path, encoding="latin-1")


def extract_first_last(df):
    return df


df = extract_first_last(df)
df.columns
df = df.drop_duplicates()
df.head()


# Define features and target
def get_X_Y(df):
    X = df.drop(columns=["id", "Target"])  # , "num", "side", "family_size"
    Y = df["Target"]
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

# Get the list of categorical column names
categorical_features = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

for x in categorical_features:
    print(X_train[x].value_counts())
