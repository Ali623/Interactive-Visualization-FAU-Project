#! Virtualenv\Scripts\python.exe

import cv2
import dlib
import joblib
import numpy as np
from sklearn.decomposition import PCA
import itertools
from keras.models import load_model

# Load the face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"archive\Cascades\shape_predictor_68_face_landmarks.dat")

# Load the saved models
classifier = load_model('nn_model.h5')

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

        # Extract X and Y coordinates for all 68 landmarks
        landmarks_list = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 68)]

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

        # Make a prediction using the SVM classifier
        prediction = classifier.predict(distances_array)
        # prediction for NN
        labels = ["anger","contempt","disgust","fear","happiness","neutrality","sadness","surprise"]
        prediction = [labels[prediction[0].argmax()]]
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
