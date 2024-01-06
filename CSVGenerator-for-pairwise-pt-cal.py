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
output_csv = "facial_landmarks_with_distances_temp_pairwise.csv"

# Initialize the CSV file for writing
with open(output_csv, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write headers to the CSV file
    header = ["Category"]
    
    # Add headers for distance columns
    for i in range(0, 50):
        header.append(f"Distance_{i}")
    
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
                        # Calculate distances between each pair of points
                        distances = []
                        for pair in pair_wise_list:
                            x1, y1 = pair[0]
                            x2, y2 = pair[1]
                            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                            distances.append(distance)

                        row.extend(distances)
                        
                        # Write the row to the CSV file
                        csv_writer.writerow(row)

print("Facial landmarks with distances have been extracted and saved to", output_csv)
