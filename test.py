import cv2
import threading
import numpy as np
from keras.models import model_from_json
from collections import defaultdict
from threading import Thread
import matplotlib.pyplot as plt
import pyttsx3
import datetime
import speech_recognition as sr
import csv
import re

# Emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create model for emotion detection
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded emotion detection model from disk")

# Initialize speech recognition engine
listener = sr.Recognizer()

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Filler words to track during interview
filler_words = ["ahh", "um", "uh", "like", "aa", "you know", "well", "actually", "basically", "seriously",
                "literally"]

# Function to capture emotions using webcam
def capture_emotions(emotion_done):
    global emotion_count, total_frames
    cap = cv2.VideoCapture(0)  # Use default camera (index 0)
    capture_duration = 60  # Duration to capture frames (in seconds)
    start_time = cv2.getTickCount()
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            emotion_label = emotion_dict[maxindex]
            emotion_count[emotion_label] += 1
        total_frames += 1
        cv2.imshow('Emotion Detection', frame)
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed_time >= capture_duration or emotion_done.is_set():
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to listen to the user's voice input and track WPM and filler words
def take_command(timeout=10):
    try:
        with sr.Microphone() as source:
            print("Listening...")
            listener.adjust_for_ambient_noise(source, duration=1)  # Adjust for ambient noise
            voice = listener.listen(source, timeout=timeout)  # Set a timeout for listening
            print("Processing...")
            command = listener.recognize_google(voice)
            print("You said:", command)
            words = command.split()
            num_words = len(words)
            duration = len(voice.frame_data) / voice.sample_rate  # Calculate duration manually
            if duration > 0:
                wpm = (num_words / duration) * 60
            else:
                wpm = 0
            filler_count = sum([command.lower().count(word) for word in filler_words])
            return command.lower(), wpm, filler_count
    except sr.WaitTimeoutError:
        print("Timeout occurred while listening.")
        return None, 0, 0
    except sr.UnknownValueError:
        print("Could not understand audio. Please speak clearly.")
        return None, 0, 0
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None, 0, 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 0, 0

# Function to read questions, sample answers, and expected keywords from CSV file
def read_questions(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return data

# Function to calculate the score based on keyword match percentage
def calculate_score(answer, expected_keywords):
    answer_keywords = set(answer.lower().split())
    expected_keywords = set(expected_keywords.lower().split())
    match_keywords = answer_keywords.intersection(expected_keywords)
    num_match_keywords = len(match_keywords)
    if len(expected_keywords) > 0:
        match_percentage = num_match_keywords / len(expected_keywords) * 100
    else:
        match_percentage = 0
    if num_match_keywords == 0:
        match_percentage = 22
    return match_percentage

# Function to convert text to speech
def talk(text):
    engine.say(text)
    engine.runAndWait()

# Function to conduct the interview
def run_interview(emotion_done):
    talk("Hello! Welcome to the interview. May I know your name?")
    print("Hello! Welcome to the interview. May I know your name?")
    name, _, _ = take_command(15)
    if name:
        talk("Nice to meet you, " + name + ". Let's start the interview.")
        print("Nice to meet you, " + name + ". Let's start the interview.")
    else:
        talk("Sorry, I couldn't recognize your name. Let's start the interview.")
    questions_data = read_questions('hr.csv')
    answers = []
    wpm_list = []
    max_questions = min(3, len(questions_data))
    for i, question_data in enumerate(questions_data[:max_questions], start=1):
        question = question_data['Question']
        print(f"Question {i}: {question}")
        talk(question)
        command, wpm, _ = take_command(20)
        wpm_list.append(wpm)
        if command:
            answers.append(command)
    if answers:
        total_score = 0
        question_scores = []
        filler_counts = []
        for answer, question_data in zip(answers, questions_data[:max_questions]):
            expected_keywords = question_data['Expected Keywords']
            score = calculate_score(answer, expected_keywords)
            total_score += score
            question_scores.append(score)
            filler_count = sum([answer.lower().count(word) for word in filler_words])
            filler_counts.append(filler_count)
        average_score = total_score / max_questions
        talk("The interview is completed. Your total score will be calculated.")
        print("The interview is completed. Your total score will be calculated.")
        total_score = sum(question_scores)
        average_score = total_score / len(question_scores)
        talk(f"Your total score is {total_score:.2f} percent.")
        talk(f"Your average score is {average_score:.2f} percent.")
        print(f"Your total score is {total_score:.2f} percent.")
        print(f"Your average score is {average_score:.2f} percent.")
        if wpm_list:
            avg_wpm = sum(wpm_list) / len(wpm_list)
            talk(f"Your average words per minute is {avg_wpm:.2f}.")
            print(f"Your average words per minute is {avg_wpm:.2f}.")
        else:
            talk("No words per minute data available.")
            print("No words per minute data available.")
        if filler_counts:
            total_fillers = sum(filler_counts)
            talk(f"You used {total_fillers} filler words in total.")
            print(f"You used {total_fillers} filler words in total.")
    else:
        talk("No answers recorded.")
        print("No answers recorded.")
    emotion_done.set()  # Set the flag to indicate that interview and emotion detection are done
    print()

# Global variables for emotion detection
emotion_count = defaultdict(int)
total_frames = 0

# Event to indicate if emotion detection is done
emotion_done = threading.Event()

# Create and start thread for emotion detection
emotion_thread = Thread(target=capture_emotions, args=(emotion_done,))
emotion_thread.start()

# Start the interview
run_interview(emotion_done)

# Wait for emotion detection thread to finish
emotion_thread.join()

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
