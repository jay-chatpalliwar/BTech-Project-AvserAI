import numpy as np
import pandas as pd
import re
from ftfy import fix_text
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import spacy
from PyPDF2 import PdfReader
import nltk
from flask import Flask, render_template, redirect, request, jsonify, session, url_for
from pyresparser import ResumeParser
import os
import time
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize app and MongoDB client
app = Flask(__name__)
app.secret_key = "your_secret_key"  # For session management
client = MongoClient("mongodb://localhost:27017/")
db = client.job_matching_app
users_collection = db.users

# NLTK setup
nltk.data.path.append("C:\\Users\\ACER/nltk_data")
nltk.download("stopwords")
stopw = set(stopwords.words("english"))

# Load job data
df = pd.read_csv("data.csv")
df["test"] = df["Job_Description"].apply(
    lambda x: " ".join(
        [word for word in str(x).split() if len(word) > 2 and word not in stopw]
    )
)

# Define senior role keywords
senior_keywords = ["senior", "staff", "lead", "manager", "director", "head", "principal", "specialist", "sr.", "expert"]

# Helper function for saving files
def save_file(file):
    upload_folder = "uploads/"
    os.makedirs(upload_folder, exist_ok=True)
    unique_filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(upload_folder, unique_filename)
    file.save(filepath)
    return filepath

# Extract text from PDF
def extract_text_from_pdf(filepath):
    text = ""
    reader = PdfReader(filepath)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Home and registration routes
@app.route("/")
def home():
    return render_template("home.html")  # Home page with options for login or signup

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        # Check if user already exists
        if users_collection.find_one({"username": username}):
            return jsonify({"error": "Username already exists"}), 400

        # Save new user
        users_collection.insert_one({"username": username, "password": password})
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = users_collection.find_one({"username": username})

        if user and check_password_hash(user["password"], password):
            session["username"] = username
            return redirect(url_for("dashboard"))
        return jsonify({"error": "Invalid credentials"}), 401
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))

# Dashboard for uploading resume
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

# Resume upload and job recommendation
@app.route("/submit_resume", methods=["POST"])
def submit_resume():
    if "username" not in session:
        return redirect(url_for("login"))

    # Handle file upload
    file = request.files["resume"]
    filepath = save_file(file)
    text = extract_text_from_pdf(filepath)

    # Parse resume and determine fresher status
    data = ResumeParser(filepath).get_extracted_data()
    resume_skills = data.get("skills", [])
    resume_experience = data.get("experience", [])
    is_fresher = not resume_experience or check_fresher(resume_experience)

    # Job matching based on resume skills
    skills_text = " ".join(resume_skills)
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform([skills_text])
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    distances, indices = nbrs.kneighbors(tfidf)

    # Filter jobs for fresher or experienced candidates
    if is_fresher:
        filtered_df = df[~df["Position"].str.contains("|".join(senior_keywords), case=False)]
    else:
        filtered_df = df

    # Rank jobs by relevance
    df_sorted = filtered_df.sort_values("match").head(9).reset_index()
    job_list = []

    for _, row in df_sorted.iterrows():
        job_list.append(
            {
                "Position": row["Position"],
                "Company": row["Company"],
                "Location": row["Location"],
                "Apply Link": row["url"],
            }
        )

    return render_template("recommendations.html", job_list=job_list)

def check_fresher(experience):
    if not experience:
        return True
    job_entries = [exp.lower() for exp in experience[:2] if isinstance(exp, str)]
    return all("intern" in entry for entry in job_entries)

def ngrams(string, n=3):
    string = fix_text(string).encode("ascii", errors="ignore").decode().lower()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = "[" + re.escape("".join(chars_to_remove)) + "]"
    string = re.sub(rx, "", string).replace("&", "and").replace(",", " ").replace("-", " ").title()
    string = re.sub(" +", " ", string).strip()
    ngrams = zip(*[string[i:] for i in range(n)])
    return ["".join(ngram) for ngram in ngrams]

if __name__ == "__main__":
    app.run(debug=True)
