import face_recognition
import cv2
import numpy as np
import pickle
from sklearn import svm
import os
import sys

encodings = []
names = []
path = 'data/'

if (len(sys.argv) > 1):
    if (sys.argv[1] == '-l'):
        train_dir = os.listdir(path)

        for person in train_dir:
            pics = os.listdir(path + person)

            # Loop through each training image for the current person
            for person_img in pics:
                print(person_img)
                # Get the face encodings for the face in each image file
                face = face_recognition.load_image_file(path + person + "/" + person_img)
                face_enc = face_recognition.face_encodings(face)[0]

                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)

        # Create and train the SVC classifier
        clf = svm.SVC(gamma='scale')
        clf.fit(encodings, names)

        # Save the trained SVM classifier
        with open('model_file', 'wb') as f:
            pickle.dump(clf, f)
else:
    with open('model_file', 'rb') as f:
        clf = pickle.load(f)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        no = len(face_locations)

        face_names = []
        # Predict all the faces in the test image using the trained classifier
        for i in range(no):
            frame_enc = face_recognition.face_encodings(rgb_small_frame, face_locations)[i]
            name = clf.predict([frame_enc])
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name[0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
