from flask import Flask, render_template, request
from waitress import serve
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from numpy.linalg import norm

app = Flask(__name__)

@app.route('/')
@app.route('/assignment')
def chatbot():
    return render_template("chatbot.html")
def calculate_document_similarity(doc_a, doc_b):
    return np.dot(doc_a, doc_b) / (norm(doc_a) * norm(doc_b))

def load_data_from_files(questions_file, answers_file):
    with open(questions_file) as f:
        questions = f.readlines()
    with open(answers_file) as f:
        answers = f.readlines()
    return questions, answers

def handle_request_information():
    questions, answers = load_data_from_files("informationRequest.txt", "answersinforequest.txt")
    vectorizer = CountVectorizer(stop_words="english")
    questions_matrix = vectorizer.fit_transform(questions)
    return questions, answers, vectorizer, questions_matrix

def handle_specific_issue():
    questions, answers = load_data_from_files("specificIssue.txt", "answersspecificissue.txt")
    vectorizer = CountVectorizer(stop_words="english")
    questions_matrix = vectorizer.fit_transform(questions)
    return questions, answers, vectorizer, questions_matrix


@app.route('/passing', methods=['POST'])
def display():
    if request.method == 'POST':
        user_enter = request.form['userenter'] 
        print('request.form:', user_enter)

        if user_enter == '1':
            questions, answers, vectorizer, questions_matrix = handle_request_information()
        elif user_enter == '2':
            questions, answers, vectorizer, questions_matrix = handle_specific_issue()
        elif user_enter == '3':
            print("A human assistant will help you shortly")
            return "A human assistant will help you shortly"
        else:
            print("Invalid input. Please select a valid option.")
            return "Invalid input. Please select a valid option."

        while True:
            user_question = input("Ask me any question you may have (type 'bye' to exit): ")
            print("User asked: " + user_question)

            if user_question.lower() == "bye":
                print("Exiting the application ...")
                break
            else:
                user_question_row = vectorizer.transform([user_question]).toarray()[0]

                if user_question_row.any():
                    max_similarity = 0
                    max_similarity_index = -1

                    for i in range(len(questions)):
                        similarity = calculate_document_similarity(questions_matrix.toarray()[i], user_question_row)
                        if (similarity > max_similarity):
                            max_similarity = similarity
                            max_similarity_index = i

                    print("Answer: " + answers[max_similarity_index])
                else:
                    print("Sorry I did not get that!")

    return render_template("chatbot.html")


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)
