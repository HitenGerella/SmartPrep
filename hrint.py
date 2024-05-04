import pyttsx3
import datetime
import speech_recognition as sr
import csv
import re

# Initialize the speech recognition engine
listener = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Filler words to track
filler_words = ["ahh","um", "uh", "like","aa", "you know", "well", "actually", "basically", "seriously", "literally"]

# Function to convert text to speech
def talk(text):
    engine.say(text)
    engine.runAndWait()

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
            
            # Calculate words per minute
            words = command.split()
            num_words = len(words)
            duration = len(voice.frame_data) / voice.sample_rate  # Calculate duration manually
            if duration > 0:
                wpm = (num_words / duration) * 60
            else:
                wpm = 0
            
            # Track filler words
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
    
    # Find the number of matching keywords
    match_keywords = answer_keywords.intersection(expected_keywords)
    num_match_keywords = len(match_keywords)
    
    # Calculate the match percentage
    if len(expected_keywords) > 0:
        match_percentage = num_match_keywords / len(expected_keywords) * 100
    else:
        match_percentage = 0  # Avoid division by zero
    
    # If there's no match, give a minimum score of 22%
    if num_match_keywords == 0:
        match_percentage = 22
        
    return match_percentage

# Function to conduct the interview
def run_interview():
    # Greet the interviewee
    talk("Hello! Welcome to the interview. May I know your name?")
    print("Hello! Welcome to the interview. May I know your name?")
    name, _, _ = take_command(15)  # Increased timeout for name recognition
    if name:
        talk("Nice to meet you, " + name + ". Let's start the interview.")
        print("Nice to meet you, " + name + ". Let's start the interview.")
    else:
        talk("Sorry, I couldn't recognize your name. Let's start the interview.")

    # Read questions, sample answers, and expected keywords from CSV file
    questions_data = read_questions('hr.csv')

    answers = []
    wpm_list = []
    max_questions = min(3, len(questions_data))  # Limit questions to 3 or the number of available questions
    for i, question_data in enumerate(questions_data[:max_questions], start=1):
        question = question_data['Question']
        
        # Present question
        print(f"Question {i}: {question}")
        talk(question)

        # Listen for the answer and track WPM
        command, wpm, _ = take_command(20)  # Adjusted timeout for answer recognition
        wpm_list.append(wpm)
        
        if command:
            answers.append(command)

    # Calculate scores and report filler words
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
            
        # Mention the total score
        average_score = total_score / max_questions
        talk("The interview is completed. Your total score will be calculated.")
        print("The interview is completed. Your total score will be calculated.")
        
        # Print the total score and average score
        total_score = sum(question_scores)
        average_score = total_score / len(question_scores)
        talk(f"Your total score is {total_score:.2f} percent.")
        talk(f"Your average score is {average_score:.2f} percent.")
        print(f"Your total score is {total_score:.2f} percent.")
        print(f"Your average score is {average_score:.2f} percent.")
        
        # Print the words per minute
        if wpm_list:
            avg_wpm = sum(wpm_list) / len(wpm_list)
            talk(f"Your average words per minute is {avg_wpm:.2f}.")
            print(f"Your average words per minute is {avg_wpm:.2f}.")
        else:
            talk("No words per minute data available.")
            print("No words per minute data available.")
        
        # Report filler words
        if filler_counts:
            total_fillers = sum(filler_counts)
            talk(f"You used {total_fillers} filler words in total.")
            print(f"You used {total_fillers} filler words in total.")
    else:
        talk("No answers recorded.")
        print("No answers recorded.")

    print()

# Start the interview
run_interview()
