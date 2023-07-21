import cv2
# import tensorflow as tf
# from tensorflow import keras
import pickle
import numpy as np
from keras import utils
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input

with open('Human_Face_Emotions.pkl', 'rb') as f:
    model = pickle.load(f)

emotions = ['angry', 'happy', 'sad']
# Set up video capture from the default camera (0)
video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
if not (video.isOpened()):
    print("Could not open video device")

# Loop through each frame from the camera
while True:
    # Read the current frame from the camera
    ret, frame = video.read()

    # Resize the frame to a fixed size (e.g., 224x224) for the model
    resized_frame = cv2.resize(frame, (224, 224))
    if not ret:
        continue  # Skip processing if the frame is empty

    # Preprocess the frame for the model
    img = image.img_to_array(resized_frame)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # Make predictions on the frame
    predictions = model.predict(img)
    predicted_emotion = emotions[np.argmax(predictions)]
    # predicted_emotion = 'angry'

    # Display the predicted emotion on the frame
    cv2.putText(frame, predicted_emotion, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the predicted emotion
    cv2.imshow('Real-time Emotion Classification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture and close the windows
video.release()
cv2.destroyAllWindows()