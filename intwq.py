import pyttsx3
import datetime
import speech_recognition as sr
import csv
import random

# Initialize the speech recognition engine
listener = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Function to convert text to speech
def talk(text):
    engine.say(text)
    engine.runAndWait()

# Function to listen to the user's voice input
def take_command():
    try:
        with sr.Microphone() as source:
            print("Listening...")
            listener.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            voice = listener.listen(source, timeout=5)  # Set a timeout for listening
            command = listener.recognize_google(voice)
            print("You said:", command)
            return command.lower()
    except sr.WaitTimeoutError:
        print("Timeout occurred while listening.")
        return None
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# Function to read questions and answers from CSV file
def read_questions(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return data

# Function to conduct the interview
def run_interview():
    # Greet the interviewee
    talk("Hello! Welcome to the interview. May I know your name?")
    name = take_command()
    if name:
        talk("Nice to meet you, " + name + ". Let's start the interview.")
    else:
        talk("Sorry, I couldn't recognize your name. Let's start the interview.")

    # Read questions and answers from CSV file
    questions_data = read_questions('ques.csv')
    
    # Randomly select five questions
    selected_questions = random.sample(questions_data, 5)

    total_marks = 0
    for question_data in selected_questions:
        question = question_data['Question']
        options = [question_data['Option A'], question_data['Option B'], question_data['Option C'], question_data['Option D']]
        expected_answer = question_data['Expected Answer']
        
        # Present question along with options
        print("Question:", question)
        print("Options are:")
        for i, option in enumerate(options, start=1):
            print(f"{chr(96+i)}: {option}")
        
        talk(question)
        talk("Options are:")
        for i, option in enumerate(options, start=1):
            talk(f"{chr(96+i)}: {option}")
        
        command = take_command()
        if command and command.strip().lower() == expected_answer.lower():
            talk('Correct answer. You got +1 mark.')
            total_marks += 1
        else:
            talk('Incorrect answer. You got 0 marks.')
    
    talk('Your total marks are ' + str(total_marks))

# Start the interview
run_interview()
