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
d_mean, d_std = ms.mean_std_of_data("facial_landmarks_with_distances_temp_pairwise.csv")

# Load the face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"archive\Cascades\shape_predictor_68_face_landmarks.dat")

# Load the saved models
classifier = load_model('nn_model_pairwise.h5')

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
        pair_wise_list = [[landmarks_list[0],landmarks_list[1]],[landmarks_list[1],landmarks_list[2]],[landmarks_list[2],landmarks_list[3]],
                          [landmarks_list[3],landmarks_list[4]],[landmarks_list[5],landmarks_list[6]],[landmarks_list[7],landmarks_list[8]],
                          [landmarks_list[8],landmarks_list[9]],[landmarks_list[10],landmarks_list[11]],[landmarks_list[11],landmarks_list[12]],
                          [landmarks_list[12],landmarks_list[13]],[landmarks_list[13],landmarks_list[16]],[landmarks_list[14],landmarks_list[15]],
                          [landmarks_list[15],landmarks_list[16]],[landmarks_list[16],landmarks_list[17]],[landmarks_list[17],landmarks_list[18]],
                          [landmarks_list[19],landmarks_list[20]],[landmarks_list[20],landmarks_list[21]],[landmarks_list[21],landmarks_list[22]],
                          [landmarks_list[22],landmarks_list[23]],[landmarks_list[23],landmarks_list[24]],[landmarks_list[24],landmarks_list[19]],
                          [landmarks_list[25],landmarks_list[26]],[landmarks_list[26],landmarks_list[27]],[landmarks_list[27],landmarks_list[28]],
                          [landmarks_list[28],landmarks_list[29]],[landmarks_list[29],landmarks_list[30]],[landmarks_list[30],landmarks_list[25]],
                          [landmarks_list[31],landmarks_list[32]],[landmarks_list[32],landmarks_list[33]],[landmarks_list[33],landmarks_list[34]],
                          [landmarks_list[34],landmarks_list[35]],[landmarks_list[35],landmarks_list[36]],[landmarks_list[36],landmarks_list[37]],
                          [landmarks_list[31],landmarks_list[42]],[landmarks_list[42],landmarks_list[41]],[landmarks_list[41],landmarks_list[40]],
                          [landmarks_list[40],landmarks_list[39]],[landmarks_list[39],landmarks_list[38]],[landmarks_list[38],landmarks_list[37]],
                          [landmarks_list[22],landmarks_list[23]],[landmarks_list[23],landmarks_list[24]],[landmarks_list[24],landmarks_list[19]],
                          [landmarks_list[43],landmarks_list[44]],[landmarks_list[44],landmarks_list[45]],[landmarks_list[45],landmarks_list[46]],
                          [landmarks_list[46],landmarks_list[47]],[landmarks_list[47],landmarks_list[48]],[landmarks_list[48],landmarks_list[43]],
                          [landmarks_list[44],landmarks_list[47]],[landmarks_list[6],landmarks_list[7]]
                          ]
        distances = []
        for pair in pair_wise_list:
            x1, y1 = pair[0]
            x2, y2 = pair[1]
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            distances.append(distance)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Convert distances to a numpy array and perform PCA
        distances_array = np.array(distances).reshape(1, -1)
        

        normalized_distances_array = (distances_array-d_mean)/d_std
        # Make a prediction using the SVM classifier
        prediction = classifier.predict(normalized_distances_array)
        # prediction for NN
        labels = ["anger","fear","happiness","neutrality","sadness","surprise"]
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
