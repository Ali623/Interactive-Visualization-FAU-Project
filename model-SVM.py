#! Virtualenv\Scripts\python.exe

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the CSV file containing facial landmarks with distances
data = pd.read_csv("facial_landmarks_with_distances.csv")

# Extract features (X) and labels (y)
X = data.drop("Category", axis=1)
y = data["Category"]

# Normalize the feature data (X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform PCA to reduce dimensionality
pca = PCA(n_components=10)  # You can adjust the number of components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train an SVM classifier
svm_classifier = SVC(kernel="linear", C=1.0)  # You can adjust the kernel and hyperparameters
svm_classifier.fit(X_train_pca, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test_pca)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average= "weighted")

print("F1 Score:", f1)
print("Accuracy:", accuracy)

# Save the models to files
joblib.dump(svm_classifier, 'svm_model.joblib')
joblib.dump(pca, 'pca_model.joblib')