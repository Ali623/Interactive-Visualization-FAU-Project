#! .venv\Scripts\python.exe

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam

# Load the CSV file containing facial landmarks with distances
data = pd.read_csv("facial_landmarks_with_distances.csv")

# Extract features (X) and labels (y)
X = data.drop("Category", axis=1)
y = data["Category"]
y = y.values

X_mean = X.mean()
X_std = X.std()
# Normalize the feature data (X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform the data
y_encoded = encoder.fit_transform(y.reshape(-1,1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Perform PCA to reduce dimensionality (adjust the number of components based on your needs)
# pca = PCA(n_components=10)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# Build a neural network
model = Sequential()
model.add(Dense(32, input_dim=1176, activation='relu'))
model.add(BatchNormalization())
 

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())


model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())


model.add(Dense(6, activation='softmax')) 

# Compile the model
optimizer = Adam(learning_rate=0.001) 
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the neural network
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
y_pred = model.predict(X_test)
y_test_decoded = [y.argmax() for y in y_test]
y_pred_decoded = [y.argmax() for y in y_pred]
accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
f1 = f1_score(y_test_decoded, y_pred_decoded, average="weighted")

print("F1 Score:", f1)
print("Accuracy:", accuracy)

# Save the model to a file
#model.save('nn_model.h5')
