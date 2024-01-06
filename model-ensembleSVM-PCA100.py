#! .venv\Scripts\python.exe

import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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

# Create a base SVM model
base_svm = SVC(kernel='linear', C=0.8)

# Create a BaggingClassifier with SVM as the base estimator
ensemble_svm = BaggingClassifier(base_svm, n_estimators=50, random_state=42)

# Train the ensemble SVM on the training data
ensemble_svm.fit(X_train, y_train)

# Make predictions on the testing data
ensemble_predictions = ensemble_svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, ensemble_predictions)
print(f'Ensemble Accuracy: {accuracy}')

# Save the models to files
joblib.dump(ensemble_svm, 'random_forest_model.joblib')
joblib.dump(pca, 'SVM_ensemble_pca100_model.joblib')