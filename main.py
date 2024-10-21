
import cv2
import os
import pickle
import numpy as np


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


faces_data = []
frames_processed = 0
samples_to_collect = 50

name = input("Enter the name of the new person: ")


if not os.path.exists('data'):
    os.makedirs('data')

print("Starting face data collection... Press 'q' to quit early.")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture video frame.")
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    
    for (x, y, w, h) in faces:
        
        face_img = frame[y:y+h, x:x+w]
        resized_image = cv2.resize(face_img, (50, 50))

        
        if len(faces_data) < samples_to_collect and frames_processed % 10 == 0:
            faces_data.append(resized_image)
        
        
        cv2.putText(frame, f"Samples: {len(faces_data)}/{samples_to_collect}", (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 255, 50), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Capture', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= samples_to_collect:
        break

    frames_processed += 1


video.release()
cv2.destroyAllWindows()


faces_data = np.array(faces_data)


faces_data_file = 'data/faces_data.pkl'
name_file = 'data/name.pkl'


if os.path.exists(faces_data_file):
    with open(faces_data_file, 'rb') as f:
        existing_faces_data = pickle.load(f)
    faces_data = np.concatenate([existing_faces_data, faces_data], axis=0)

with open(faces_data_file, 'wb') as f:
    pickle.dump(faces_data, f)


if os.path.exists(name_file):
    with open(name_file, 'rb') as f:
        existing_names = pickle.load(f)
    names = existing_names + [name] * samples_to_collect
else:
    names = [name] * samples_to_collect

with open(name_file, 'wb') as f:
    pickle.dump(names, f)

print(f"Face data collection complete. {samples_to_collect} samples saved for {name}.")
