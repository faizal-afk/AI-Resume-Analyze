from flask import Flask, render_template, request
import pdfplumber
from docx import Document
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

roles = {

    "AI Engineer": [
        "Python","Machine Learning","Deep Learning",
        "NLP","TensorFlow","PyTorch","Git"
    ],

    "Data Scientist": [
        "Python","Machine Learning","Statistics",
        "Pandas","NumPy","SQL","Data Visualization"
    ],

    "Data Analyst": [
        "Python","SQL","Excel",
        "Power BI","Pandas","Data Analysis"
    ],

    "Machine Learning Engineer": [
        "Python","Machine Learning",
        "Scikit-learn","TensorFlow",
        "PyTorch","MLOps"
    ],

    "Java Developer": [
        "Java","OOP","Spring Boot",
        "MySQL","REST API","Git"
    ],

    "Python Developer": [
        "Python","Flask","Django",
        "REST API","SQL","Git"
    ],

    "Web Developer": [
        "HTML","CSS","JavaScript",
        "React","Git","Bootstrap"
    ],

    "Frontend Developer": [
        "HTML","CSS","JavaScript",
        "React","Bootstrap","Git"
    ],

    "Backend Developer": [
        "Python","Django","Flask",
        "SQL","REST API","Git"
    ],

    "Full Stack Developer": [
        "HTML","CSS","JavaScript",
        "React","Python","Flask",
        "SQL","Git"
    ],

    "Cloud Engineer": [
        "AWS","Azure","Docker",
        "Linux","Git","Networking"
    ],

    "DevOps Engineer": [
        "Docker","Kubernetes",
        "Linux","Git","CI/CD"
    ],

    "Cyber Security Analyst": [
        "Network Security","Linux",
        "Penetration Testing",
        "Cryptography","Python"
    ],

    "Software Engineer": [
        "Java","Python",
        "Data Structures",
        "Algorithms","OOP","Git"
    ]
}

def extract_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    return text.lower()

def extract_docx_text(path):
    doc = Document(path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + " "
    return text.lower()

@app.route("/", methods=["GET", "POST"])
def home():

    score = None
    rating = None
    found = []
    missing = []
    suggestions = []
    selected_role = None

    if request.method == "POST":

        selected_role = request.form["role"]
        file = request.files["resume"]

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        if file.filename.endswith(".pdf"):
            text = extract_pdf_text(filepath)

        elif file.filename.endswith(".docx"):
            text = extract_docx_text(filepath)

        else:
            return render_template(
                "index.html",
                error="Only PDF and DOCX files allowed",
                roles=roles.keys()
            )

        required_skills = roles[selected_role]

        for skill in required_skills:

            if skill.lower() in text:
                found.append(skill)

            else:
                missing.append(skill)

        score = round(
            (len(found) / len(required_skills)) * 100,
            2
        )

        docs = [
            text,
            " ".join(required_skills)
        ]

        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(docs)

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        similarity_score = round(
            similarity * 100,
            2
        )

        if score >= 80:
            rating = "Excellent"
        elif score >= 60:
            rating = "Good"
        elif score >= 40:
            rating = "Average"
        else:
            rating = "Needs Improvement"

        for skill in missing:
            suggestions.append(
                f"Add or learn {skill}"
            )

        return render_template(
            "index.html",
            roles=roles.keys(),
            score=score,
            rating=rating,
            similarity=similarity_score,
            found=found,
            missing=missing,
            suggestions=suggestions,
            selected_role=selected_role
        )

    return render_template(
        "index.html",
        roles=roles.keys()
    )

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)