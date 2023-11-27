#! .venv\Scripts\python.exe

import dlib
import cv2
import os
import csv
import itertools  # Import itertools to generate pairs of points

# Load the face detector and facial landmarks predictor
face_cascade = cv2.CascadeClassifier(r"archive\Cascades\haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor(r"archive\Cascades\shape_predictor_68_face_landmarks.dat")

# Root folder with subfolders as categories
root_folder = r"archive\fer_ckplus_kdef"
output_csv = "facial_landmarks_with_distances.csv"

# Initialize the CSV file for writing
with open(output_csv, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write headers to the CSV file
    header = ["Category"]
    
    # Add headers for distance columns
    for i in range(0,49):
        for j in range(i + 1, 49):
            header.append(f"Distance_{i}_{j}")
    
    csv_writer.writerow(header)
    
    # Process each subfolder in the root folder
    for category in os.listdir(root_folder):
        category_path = os.path.join(root_folder, category)
        
        if os.path.isdir(category_path):
            # Process each image in the subfolder
            for filename in os.listdir(category_path):
                if filename.endswith(".png"):
                    image_path = os.path.join(category_path, filename)
                    image = cv2.imread(image_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(0,0))
                    for (x,y,w,h) in faces:
                        face = dlib.rectangle(x, y, x + w, y + h)
                        # Extracting Region of interest
                        face_roi = gray[y:y + h, x:x + w]
                        landmarks = predictor(face_roi, face)
                        
                        # Initialize a row for the current image's landmarks
                        row = [category]
                        landmark_points = [num for num in range(17, 68) if num not in (60, 64)]
                        # Extract X and Y coordinates for all 68 landmarks
                        landmarks_list = [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_points]
                        
                        # Calculate distances between each pair of points
                        distances = []
                        for pair in itertools.combinations(landmarks_list, 2):
                            x1, y1 = pair[0]
                            x2, y2 = pair[1]
                            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                            distances.append(distance)

                        row.extend(distances)
                        
                        # Write the row to the CSV file
                        csv_writer.writerow(row)

print("Facial landmarks with distances have been extracted and saved to", output_csv)
