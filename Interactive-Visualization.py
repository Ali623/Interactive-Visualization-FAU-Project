#! .venv\Scripts\python.exe

import cv2
import dlib
import joblib
import numpy as np
from sklearn.decomposition import PCA
import itertools
from keras.models import load_model
import mean_std_ofdata as ms

# find mean and std of the dataset to normalize the input data
d_mean, d_std = ms.mean_std_of_data("facial_landmarks_with_distances.csv")

# Load the face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"archive\Cascades\shape_predictor_68_face_landmarks.dat")

# Load the saved models
#classifier = joblib.load('svm_model.joblib')
#pca = joblib.load('pca_model.joblib')

# Load the saved models
classifier = joblib.load('random_forest_model.joblib')
pca = joblib.load('pca_model.joblib')


# Initialize the camera for real-time video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Process each detected face
    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(gray, face)

        landmark_points = [num for num in range(17, 68) if num not in (60, 64)]
        # Extract X and Y coordinates for all 68 landmarks
        landmarks_list = [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_points]

        # Draw points on the face
        for (x, y) in landmarks_list:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Extract the region of interest (ROI) for the detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = frame[y:y+h, x:x+w]

        # Calculate distances between each pair of points
        distances = []
        for pair in itertools.combinations(landmarks_list, 2):
            x1, y1 = pair[0]
            x2, y2 = pair[1]
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            distances.append(distance)
            #cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Convert distances to a numpy array and perform PCA
        distances_array = np.array(distances).reshape(1, -1)
        
        normalized_distances_array = (distances_array-d_mean)/d_std
        #distances_normalized = scaler.fit_transform(distances_array)
        distances_pca = pca.transform(normalized_distances_array)

        # Make a prediction using the SVM classifier
        prediction = classifier.predict(distances_pca)
        # Display the prediction on the frame
        cv2.putText(frame, f"Prediction: {prediction[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # Display the frame with landmarks
    cv2.imshow('Real-time Face Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
