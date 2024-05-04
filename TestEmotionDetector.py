import cv2
import numpy as np
from keras.models import model_from_json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Start the webcam feed
cap = cv2.VideoCapture(0)  # Use default camera (index 0)

# Variables to keep track of detected emotions
emotion_count = defaultdict(int)
total_frames = 0

# Time duration to capture frames (in seconds)
capture_duration = 60
start_time = cv2.getTickCount()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in num_faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        
        # Preprocess face image
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[maxindex]
        
        # Increment count for detected emotion
        emotion_count[emotion_label] += 1
    
    total_frames += 1

    # Display frame with emotion detection
    cv2.imshow('Emotion Detection', frame)
    
    # Calculate elapsed time
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    
    # Check if 60 seconds have elapsed
    if elapsed_time >= capture_duration:
        break
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Calculate the percentage of each detected emotion
emotions = []
percentages = []
for emotion, count in emotion_count.items():
    percentage = (count / total_frames) * 100
    emotions.append(emotion)
    percentages.append(percentage)

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(emotions, percentages, color='skyblue')
plt.xlabel('Emotion')
plt.ylabel('Percentage')
plt.title('Emotion Detection Results')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
