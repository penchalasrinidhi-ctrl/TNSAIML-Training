import numpy as np  # Used for numerical operations
import pandas as pd  # Used to load and manipulate tabular data

# Import Logistic Regression, a classification algorithm
from sklearn.ensemble import RandomForestClassifier

# Import function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import evaluation metrics to measure model performance
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)

# Path helps locate files and directories
from pathlib import Path

# Joblib is used to save the trained model and feature names
import joblib


# --------------------------------------------------
# STEP 2: READ THE CSV DATASET
# --------------------------------------------------

# Define the path to the credit card fraud dataset.
# Update this path to match the actual location of your file.
data = Path("C:\\Users\\Dell\\OneDrive\\Desktop\\Credit_Card_Fraud_Detection\\dataset\\creditcard.csv")

# Read the CSV file and store the data in a Pandas DataFrame
df = pd.read_csv(data)


# --------------------------------------------------
# STEP 3: PERFORM EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------

# Display the number of rows and columns in the dataset
print("Dataset shape:")
print(df.shape)

# Display column names, data types, and non-null value counts
print("\nDataset information:")
df.info()

# Display statistical summaries of numerical columns
# Includes count, mean, standard deviation, min, max, and quartiles
print("\nStatistical summary:")
print(df.describe())

# Count missing values in each column
print("\nMissing values in each column:")
print(df.isnull().sum())

# Count duplicate rows in the dataset
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# --------------------------------------------------
# STEP 4: SEPARATE FEATURES AND TARGET
# --------------------------------------------------

# Features (X) are the input columns used to make predictions.
# Remove the target column "Class" from the dataset.
X = df.drop("Class", axis=1)

# Target (y) is the column the model needs to predict.
y = df["Class"]


# --------------------------------------------------
# STEP 5: SPLIT DATA INTO TRAINING AND TESTING SETS
# --------------------------------------------------

# Split the dataset into:
# 80% training data for learning
# 20% testing data for evaluating performance

# random_state=42 ensures reproducible results.
# stratify=y preserves the target-class proportions in both the training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# STEP 6: TRAIN THE RANDOM FOREST MODEL
# --------------------------------------------------

# Create the Random Forest classifier.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training features and labels.
model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# --------------------------------------------------
# STEP 7: TEST AND EVALUATE THE MODEL
# --------------------------------------------------

# Predict target labels for the unseen testing data.
y_test_pred = model.predict(X_test)

# Calculate accuracy: proportion of correct predictions
accuracy = accuracy_score(y_test, y_test_pred)

# Calculate precision: proportion of positive predictions that are actually positive
precision = precision_score(
    y_test,
    y_test_pred,
    pos_label=1,
    zero_division=0
)

# Calculate recall: proportion of actual positive cases correctly identified by the model
recall = recall_score(
    y_test,
    y_test_pred,
    pos_label=1,
    zero_division=0
)

# Calculate F1 score: harmonic mean of precision and recall
f1 = f1_score(
    y_test,
    y_test_pred,
    pos_label=1,
    zero_division=0
)

# Display the evaluation metrics
print("\nModel Evaluation Results")
print("------------------------")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")


# --------------------------------------------------
# STEP 8: SAVE THE TRAINED MODEL AND FEATURE NAMES
# --------------------------------------------------

# Save the trained Random Forest model to a file.
# The Streamlit application will load this file later.
joblib.dump(model, "model.joblib")

# Save the feature names in the same order used for training.
# This helps the Streamlit app arrange user inputs correctly.
joblib.dump(list(X.columns), "feature_names.joblib")

print("\nModel saved as model.joblib")
print("Feature names saved as feature_names.joblib")
