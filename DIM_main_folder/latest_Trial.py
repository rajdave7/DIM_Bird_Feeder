import cv2
import tensorflow as tf
import numpy as np

# Load the trained bird classification model
model = tf.keras.models.load_model('birds_classification_vgg.h5')

# Load the Haar Cascade classifier for face detection
cascade_classifier = cv2.CascadeClassifier(
    'DIM\haarcascade_frontalface_default.xml')

# Start the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video feed
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bird faces in the grayscale image
    bird_faces = cascade_classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # For each bird face, classify the bird species using the trained model
    for (x, y, w, h) in bird_faces:
        # Extract the bird face from the frame
        bird_face = frame[y:y+h, x:x+w]

        # Resize the bird face to match the input size of the model
        bird_face = cv2.resize(bird_face, (224, 224))

        # Normalize the pixel values to be between 0 and 1
        bird_face = bird_face / 255.0

        # Add an extra dimension to the image to represent the batch size (1)
        bird_face = np.expand_dims(bird_face, axis=0)

        # Classify the bird species using the trained model
        predictions = model.predict(bird_face)

        # Get the index of the predicted bird species
        bird_index = np.argmax(predictions)

        # Map the bird index to the corresponding bird species name
        if bird_index == 0:
            bird_name = 'crow'
        elif bird_index == 1:
            bird_name = 'pigeon'
        elif bird_index == 2:
            bird_name = 'sparrow'
        else:
            bird_name = 'unknown'

        # Draw a rectangle around the bird face and label it with the predicted bird species name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, bird_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Bird Detector', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
