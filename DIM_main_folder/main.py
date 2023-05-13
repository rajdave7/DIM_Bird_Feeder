import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt

# Load the pretrained model
model = load_model('birds_classification_vgg.h5')

# Define the bird classes
bird_classes = ['woodpecker', 'golden eagle', 'sparrow', 'bulbul', 'pigeon', ]

# Set up the webcam
cap = cv2.VideoCapture(1)
ret, frame = cap.read()


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Preprocess the image
    img = cv2.resize(frame, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0

    # Run inference
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    bird_class = bird_classes[class_idx]

    # Check if bird is detected
    if bird_class != 'none':
        # Display the results
        cv2.putText(frame, bird_class, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Bird Classification', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
