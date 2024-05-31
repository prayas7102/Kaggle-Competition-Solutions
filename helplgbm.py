import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from lightgbm import LGBMClassifier
import pickle

# Load data
df = pd.read_csv("./train.csv", encoding="latin-1")

# Feature extraction
def extract_first_last(df):
    df[['deck', 'num', 'side']] = df['Cabin'].str.split('/', expand=True)
    df['group'] = df['PassengerId'].str[:4]
    df['family_size'] = [list(df['group']).count(x) for x in list(df['group'])]
    return df

df = extract_first_last(df)
df = df.drop_duplicates()

# Define features and target
def get_X_Y(df):
    X = df.drop(columns=["PassengerId", "Name", "Transported", "Cabin", "group", "num"])
    Y = df["Transported"]
    return X, Y

X, Y = get_X_Y(df)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Preprocessing pipelines
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer()),
    ('scaler', StandardScaler()),
    ('power', PowerTransformer())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
)

# Define the model
model = LGBMClassifier()

# Define the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Hyperparameter tuning
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.3],
    'model__max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, Y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# Save the best model
best_pipeline = grid_search.best_estimator_
filename_pkl = "best_model.pkl"
pickle.dump(best_pipeline, open(filename_pkl, "wb"))
print(f"Model saved as {filename_pkl}")

# Evaluate the best model
y_pred = best_pipeline.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(Y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(Y_test, y_pred)}")

# Cross-validation
cv_scores = cross_val_score(best_pipeline, X_train, Y_train, cv=10, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean()}")

# Generate submission
def generate_submission(test_file):
    df = pd.read_csv(test_file)
    df.replace('', np.nan, inplace=True)
    df = extract_first_last(df)
    filtered_df = df[X_train.columns]
    predictions = best_pipeline.predict(filtered_df)
    original_df = pd.read_csv(test_file)
    original_df['Transported'] = predictions
    submission_df = original_df[['PassengerId', 'Transported']]
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file saved as 'submission.csv'")

# Generate the submission
test_file = 'test.csv'
generate_submission(test_file)
