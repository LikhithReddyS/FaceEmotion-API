import cv2
from keras.models import load_model
import numpy as np

# Load the trained emotion detection model
model = load_model("emotiondetector.h5")

# Load the face detection model (Haar Cascade for face detection)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize webcam (0 is typically the default camera)
webcam = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not webcam.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Define the function to extract features (replace with your actual feature extraction function)
def extract_features(image):
    image = image.astype('float32') / 255.0  # Normalize image to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Expand dims for channel (grayscale)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Labels for the emotions (ensure these match your model's outputs)
labels = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

while True:
    ret, im = webcam.read()  # Capture a frame from the webcam
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = face_cascade.detectMultiScale(im, 1.3, 5)  # Detect faces in the frame
    
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]  # Extract the face region
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)  # Draw a rectangle around the face
            image = cv2.resize(image, (48, 48))  # Resize face to match input size for model
            img = extract_features(image)  # Feature extraction
            pred = model.predict(img)  # Get model prediction
            prediction_label = labels[pred.argmax()]  # Get the predicted label
            
            # Display the prediction on the image
            cv2.putText(im, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255)) 
        
        cv2.imshow("Output", im)  # Display the frame with the detected face and prediction
        if cv2.waitKey(27) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    except cv2.error:
        pass

# Release webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
