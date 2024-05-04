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
def take_command(timeout=5):
    try:
        with sr.Microphone() as source:
            print("Listening...")
            listener.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            voice = listener.listen(source, timeout=timeout)  # Set a timeout for listening
            print("Processing...")
            command = listener.recognize_google(voice)
            print("You said:", command)
            return command.lower()
    except sr.WaitTimeoutError:
        print("Timeout occurred while listening.")
        return None
    except sr.UnknownValueError:
        print("Could not understand audio. Please speak clearly.")
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
    name = take_command(10)  # Increased timeout for name recognition
    if name:
        talk("Nice to meet you, " + name + ". Let's start the interview.")
    else:
        talk("Sorry, I couldn't recognize your name. Let's start the interview.")

    # Read questions and answers from CSV file
    questions_data = read_questions('ques2.csv')
    
    # Randomly select five questions
    selected_questions = random.sample(questions_data, 5)

    total_marks = 0
    correct_answers = []  # Store correct answers for printing at the end
    for i, question_data in enumerate(selected_questions, start=1):
        question = question_data['Question']
        expected_answer = question_data['Expected Answer']
        
        # Present question
        print(f"Question {i}: {question}")
        talk(question)

        # Extract keywords from expected answer
        keywords = [word.lower() for word in expected_answer.split()]
        
        command = take_command(7)  # Adjusted timeout for answer recognition
        if command and any(keyword in command.lower() for keyword in keywords):
            total_marks += 1
            correct_answers.append((question, expected_answer))

    # Mention the score
    talk(f'Your total score is {total_marks} out of 5.')
    print(f'Your total score is {total_marks} out of 5.')
    print()

    # # Print correct answers at the end
    # talk('The correct answers are:')
    # print('The correct answers are:')
    # for i, answer in enumerate(correct_answers, start=1):
    #     question, expected_answer = answer
    #     print(f'Question {i}: {expected_answer}')
    #     talk(f'Question {i}: {expected_answer}')
        
    print()

    # Print all expected answers
    talk('Here are the expected answers to all the questions:')
    print('Expected answers to all the questions:')
    for i, question_data in enumerate(selected_questions, start=1):
        expected_answer = question_data['Expected Answer']
        print(f'Question {i}: {expected_answer}')
        talk(f'Question {i}: {expected_answer}')

# Start the interview
run_interview()
