#! .venv\Scripts\python.exe

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA

# Load the CSV file containing facial landmarks with distances
data = pd.read_csv("facial_landmarks_with_distances.csv")

# Extract features (X) and labels (y)
X = data.drop("Category", axis=1)
y = data["Category"]
y = y.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform PCA to reduce dimensionality (adjust the number of components based on your needs)
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Standardize the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Support Vector Machine classifier
svm_classifier = SVC(kernel='linear', C = 0.5)

# Train the classifier on the training data and collect training history
history = svm_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
f1_sco = f1_score(y_test, y_pred, average="weighted")
print(f'Accuracy: {accuracy}')
print(f'f1_Score: {f1_sco}')

# Save the models to files
joblib.dump(svm_classifier, 'svm_model.joblib')
joblib.dump(pca, 'SVM_pca100_model.joblib')