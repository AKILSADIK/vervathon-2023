import os
import PyPDF2
import pprint
from lmqg import TransformersQG
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

model = TransformersQG('lmqg/t5-base-squad-qag')


@app.route("/", methods=["POST","GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["demo1"]
    video_path = "uploaded_pdf.pdf"
    file.save(video_path)

    # Open the PDF file
    with open('uploaded_pdf.pdf', 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)

        # Extract text from all pages
        context = ''
        for page in pdf_reader.pages:
            context += page.extract_text()

    # Split the input text into multiple parts if it exceeds the maximum length
    max_length = 512
    if len(model.tokenizer.encode(context)) > max_length:
        split_context = []
        start_index = 0
        end_index = max_length
        while end_index <= len(context):
            split_context.append(context[start_index:end_index])
            start_index = end_index
            end_index += max_length
    else:
        split_context = [context]

    # Generate questions for each part of the input text
    questions_list = []
    for part in split_context:
        question_answer = model.generate_qa(part)
        questions = [question for question, _ in question_answer]
        questions_list.extend(questions)

    # Print the generated questions
    for question in questions_list:
        pprint.pprint(question)

    return questions_list

if __name__ == "__main__":
    app.run(debug=True)