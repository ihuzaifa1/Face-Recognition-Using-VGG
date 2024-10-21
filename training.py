import cv2
import os
import numpy as np
import tensorflow as tf
import pickle
import csv
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from datetime import datetime


data_dir = 'data/'
attendance_dir = 'attendance/'
model_path = 'face_recognition_model.h5'
faces_data_path = os.path.join(data_dir, 'faces_data.pkl')
labels_path = os.path.join(data_dir, 'name.pkl')
encoder_path = os.path.join(data_dir, 'label_encoder.pkl')


if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)


if os.path.exists(model_path):
    
    print("Loading pre-trained model...")
    model = load_model(model_path)
    
    
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)

else:
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)

    
    with open(faces_data_path, 'rb') as f:
        faces_data = pickle.load(f)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)

    
    faces_data = np.array(faces_data)
    labels = np.array(labels)

    
    faces_data_resized = np.array([cv2.resize(face, (224, 224)) for face in faces_data])

    
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)

    
    num_classes = len(np.unique(labels))
    output = Dense(num_classes, activation='softmax')(x)

    
    model = Model(inputs=base_model.input, outputs=output)

    
    for layer in base_model.layers:
        layer.trainable = False

    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    
    labels = to_categorical(labels, num_classes=num_classes)

    
    X_train, X_test, y_train, y_test = train_test_split(faces_data_resized, labels, test_size=0.2, random_state=42)

    
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    
    model.save(model_path)


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
last_recognized_face = None  
attended_faces = set()  

while True:
    ret, frame = video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img.reshape(1, 224, 224, 3)
        prediction = model.predict(face_img)
        class_id = np.argmax(prediction)
        label = le.inverse_transform([class_id])[0]
        
        
        last_recognized_face = label

       
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Face Recognition', frame)
    
   
    if cv2.waitKey(1) & 0xFF == ord('n'):
        if last_recognized_face and last_recognized_face not in attended_faces:  
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            attendance_file = os.path.join(attendance_dir, f'attendance_{datetime.now().strftime("%Y%m%d")}.csv')
            
            with open(attendance_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, last_recognized_face])  
            print(f"Attendance saved for {last_recognized_face}: {attendance_file}")
            attended_faces.add(last_recognized_face)  

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

