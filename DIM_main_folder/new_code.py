import cv2
import tensorflow as tf
import numpy as np

# Load bird detection cascades
bird1_cascade = cv2.CascadeClassifier('bird1-cascade.xml')
bird2_cascade = cv2.CascadeClassifier('bird2-cascade.xml')

# Load bird species classification model
model = tf.keras.models.load_model('birds_classification_vgg.h5')

# Define a function to get the name of the predicted bird species


def get_species_name(pred):
    # Load class names
    class_names = ['bulbul', 'woodpecker',
                   'golden eagle', 'sparrow', 'pigeon', ]
    # Get index of the predicted class
    class_idx = np.argmax(pred)
    # Get the name of the predicted class
    species_name = class_names[class_idx]
    return species_name


# Open video capture device
cap = cv2.VideoCapture(1)

while True:
    # Read frame from the video capture device
    ret, img = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect birds using the two cascades
    birds1 = bird1_cascade.detectMultiScale(gray, 1.3, 5)
    birds2 = bird2_cascade.detectMultiScale(gray, 1.3, 5)

    # Process birds detected by cascade 1
    for (x, y, w, h) in birds1:
        # Draw bounding box around the bird
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Extract bird image from the frame
        bird_img = gray[y:y+h, x:x+w]

        # Resize bird image to match the input size of the species classification model
        bird_img = cv2.resize(bird_img, (224, 224))

        # Preprocess the bird image for the species classification model
        bird_img = bird_img.astype('float32') / 255.0
        bird_img = tf.expand_dims(bird_img, 0)

        # Predict the species of the bird using the classification model
        species_pred = model.predict(bird_img)[0]

        # Get the name of the predicted species
        species_name = get_species_name(species_pred)

        # Draw species name on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, species_name, (x, y-10), font,
                    0.5, (0, 255, 255), 2, cv2.LINE_AA)

    # Process birds detected by cascade 2
    for (x, y, w, h) in birds2:
        # Draw bounding box around the bird
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Extract bird image from the frame
        bird_img = gray[y:y+h, x:x+w]

        # Resize bird image to match the input size of the species classification model
        bird_img = cv2.resize(bird_img, (224, 224))

        # Preprocess the bird image for the species classification model
        bird_img = bird_img.astype('float32') / 255.0
        bird_img = tf.expand_dims(bird_img, 0)

        # Predict the species of the bird using the classification model
        species_pred = model.predict(bird_img)[0]

        # Get the name of the predicted species
        species_name = get_species_name(species_pred)

        # Draw species name on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, species_name, (x, y-10), font,
                    0.5, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('img', img)

    # Check for exit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture device and close all windows
cap.release()
cv2.destroyAllWindows()
