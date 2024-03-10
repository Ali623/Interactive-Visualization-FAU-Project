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
classifier = load_model('nn_model.h5')

# Path to the image you want to predict
image_path = r"picture\Su.png"

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    # for (x, y) in landmarks_list:
    #     cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # Extract the region of interest (ROI) for the detected face
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_roi = img[y:y+h, x:x+w]

    # Calculate distances between each pair of points
    distances = []
    for pair in itertools.combinations(landmarks_list, 2):
        x1, y1 = pair[0]
        x2, y2 = pair[1]
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        distances.append(distance)
        #cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Convert distances to a numpy array and perform PCA
    distances_array = np.array(distances).reshape(1, -1)
    

    normalized_distances_array = (distances_array-d_mean)/d_std
    # Make a prediction using the SVM classifier
    prediction = classifier.predict(normalized_distances_array)
    # prediction for NN
    labels = ["anger","fear","happiness","neutrality","sadness","surprise"]
    # prediction_percentage = int(prediction[0].max() * 100)
    prediction = [labels[prediction[0].argmax()]]
    # Display the prediction on the frame
    # cv2.putText(img, f"Prediction: {prediction[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Display the prediction on the frame with reduced font size
    text = f"Prediction: {prediction[0]}" #-{prediction_percentage}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Adjust the font scale as needed
    font_thickness = 1  # Adjust the font thickness as needed
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = x - 20  # Position of the text on the x-axis
    text_y = y - 10  # Position of the text slightly above the rectangle
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Draw a rectangle around the detected face
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Adjust rectangle color and thickness as needed

# Save the resulting image
output_image_path = "Su-pred-2.png"
cv2.imwrite(output_image_path, img)

# Display the img with landmarks
cv2.imshow('Real-time Face Recognition', img)


# Display the image for a certain amount of time (e.g., 5 seconds)
cv2.waitKey(5000)

# Close all OpenCV windows
cv2.destroyAllWindows()