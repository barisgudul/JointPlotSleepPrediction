# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv("msleep.csv")
subset = dataset.iloc[:, 6:11]

# Check and clean missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(subset)
subset = imputer.transform(subset)
dataset.iloc[:, 6:11] = subset

# Define features and target variable
X = pd.concat([dataset.iloc[:, 2:3], dataset.iloc[:, 6:11]], axis=1).values
y = dataset.iloc[:, 5].values

# Transform categorical data (One-Hot Encoding)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions with the test data
y_pred = regressor.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), axis=1))

# Calculate the error rate
error = np.abs(y_test - y_pred)

# Combine predictions and error rate into a single DataFrame
df = pd.DataFrame({
    "Predicted Sleep": y_pred,
    "Error": error
})

# Plot joint plot
plt.figure(figsize=(15, 8))
sns.jointplot(x="Predicted Sleep", y="Error", data=df, kind="reg")
plt.show()
