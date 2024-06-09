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

for x in df.columns:
    print(df[x].value_counts())












# Given your dataset, here are some preprocessing steps you can take to improve the accuracy of your model:

# ### 1. Handle Imbalanced Classes
# The target variable (`Target`) shows an imbalance. Techniques to handle this include:

# - **Resampling:** You can either oversample the minority class or undersample the majority class.
  
#   ```python
#   from imblearn.over_sampling import SMOTE
#   from imblearn.under_sampling import RandomUnderSampler
#   from imblearn.combine import SMOTETomek

#   smote = SMOTE(random_state=42)
#   X_res, y_res = smote.fit_resample(X, y)
#   ```

# - **Class Weight Adjustment:** Adjust the class weights in your model to give more importance to minority classes.

#   ```python
#   model = GradientBoostingClassifier(random_state=42, class_weight='balanced')
#   ```

# ### 2. Encode Categorical Variables
# Many of your features are categorical (e.g., `Marital status`, `Application mode`). Convert them into numerical values.

# - **Label Encoding:** For ordinal categories.

#   ```python
#   from sklearn.preprocessing import LabelEncoder

#   label_encoder = LabelEncoder()
#   data['Marital status'] = label_encoder.fit_transform(data['Marital status'])
#   ```

# - **One-Hot Encoding:** For nominal categories.

#   ```python
#   data = pd.get_dummies(data, columns=['Application mode'])
#   ```

# ### 3. Handle Missing Values
# Ensure there are no missing values in your dataset.

# - **Simple Imputer:** Fill missing values with mean, median, or most frequent values.

#   ```python
#   from sklearn.impute import SimpleImputer

#   imputer = SimpleImputer(strategy='mean')
#   data['Previous qualification (grade)'] = imputer.fit_transform(data[['Previous qualification (grade)']])
#   ```

# ### 4. Feature Scaling
# Scale the features to normalize the data.

# - **Standard Scaling:** Scale features to have zero mean and unit variance.

#   ```python
#   from sklearn.preprocessing import StandardScaler

#   scaler = StandardScaler()
#   data[['Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']] = scaler.fit_transform(data[['Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']])
#   ```

# ### 5. Feature Engineering
# Create new features or transform existing ones to capture more information.

# - **Binning Continuous Variables:** Create bins for continuous variables to capture non-linear relationships.

#   ```python
#   data['Age at enrollment_bin'] = pd.cut(data['Age at enrollment'], bins=5, labels=False)
#   ```

# ### 6. Removing Outliers
# Identify and remove outliers using methods such as Z-Score or IQR.

# - **Z-Score Method:**

#   ```python
#   from scipy import stats
#   import numpy as np

#   z_scores = np.abs(stats.zscore(data[['Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']]))
#   data = data[(z_scores < 3).all(axis=1)]
#   ```

# - **IQR Method:**

#   ```python
#   Q1 = data['Admission grade'].quantile(0.25)
#   Q3 = data['Admission grade'].quantile(0.75)
#   IQR = Q3 - Q1
#   data = data[(data['Admission grade'] >= (Q1 - 1.5 * IQR)) & (data['Admission grade'] <= (Q3 + 1.5 * IQR))]
#   ```

# ### 7. Correlation Analysis
# Analyze the correlation between features and remove highly correlated ones to reduce multicollinearity.

# ```python
# import seaborn as sns
# import matplotlib.pyplot as plt

# corr_matrix = data.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.show()

# # Drop highly correlated features if necessary
# ```

# ### Example Workflow

# Here's an example workflow incorporating some of these steps:

# ```python
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.impute import SimpleImputer
# from imblearn.over_sampling import SMOTE
# from scipy import stats
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load your dataset
# data = pd.read_csv('your_dataset.csv')

# # Encode categorical variables
# label_encoder = LabelEncoder()
# data['Marital status'] = label_encoder.fit_transform(data['Marital status'])
# data = pd.get_dummies(data, columns=['Application mode'])

# # Handle missing values
# imputer = SimpleImputer(strategy='mean')
# data['Previous qualification (grade)'] = imputer.fit_transform(data[['Previous qualification (grade)']])

# # Remove outliers using Z-Score method
# z_scores = np.abs(stats.zscore(data[['Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']]))
# data = data[(z_scores < 3).all(axis=1)]

# # Scale features
# scaler = StandardScaler()
# data[['Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']] = scaler.fit_transform(data[['Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']])

# # Handle imbalanced classes using SMOTE
# X = data.drop('Target', axis=1)
# y = data['Target']
# smote = SMOTE(random_state=42)
# X_res, y_res = smote.fit_resample(X, y)

# # Correlation analysis
# corr_matrix = data.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.show()

# # Model training and evaluation can follow here
# ```

# Adjust the parameters and methods according to your specific dataset and problem. This workflow is a starting point to improve your data preprocessing and model accuracy.