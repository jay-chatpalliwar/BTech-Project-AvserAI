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
from flask import Flask, render_template, redirect, request, jsonify, session
from pyresparser import ResumeParser
import os
import time

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secret_key"  # Replace with a strong secret key

# Data processing
df = pd.read_csv("data.csv")
stopw = set(stopwords.words("english"))
df["test"] = df["Job_Description"].apply(lambda x: " ".join([word for word in str(x).split() if len(word) > 2 and word not in stopw]))

# Initialize models and configurations
vectorizer = TfidfVectorizer(min_df=1, analyzer='word', lowercase=True)
tfidf = vectorizer.fit_transform(df["test"])
nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(tfidf)

senior_keywords = ["senior", "lead", "manager", "director", "head", "sr.", "expert"]

# Helper function to save files
def save_file(file):
    upload_folder = "uploads/"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    unique_filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(upload_folder, unique_filename)
    file.save(filepath)
    return filepath

# Helper function to extract text from PDF
def extract_text_from_pdf(filepath):
    text = ""
    reader = PdfReader(filepath)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Check if a candidate is a fresher
def check_fresher(experience):
    if not experience:
        return True
    has_intern = any("intern" in exp.lower() for exp in experience[:2] if isinstance(exp, str))
    return has_intern

# Content-based recommendation function
def content_based_recommendation(resume_skills):
    resume_tfidf = vectorizer.transform([" ".join(resume_skills)])
    distances, indices = nbrs.kneighbors(resume_tfidf)
    return [(df.iloc[i], 1 - distances[0][idx]) for idx, i in enumerate(indices[0])]

# Collaborative filtering recommendation (mocked with placeholder data for user interest)
def collaborative_recommendation(user_skills):
    similar_users_jobs = ["Data Scientist", "Machine Learning Engineer"]  # Replace with collaborative data
    collaborative_jobs = df[df["Position"].isin(similar_users_jobs)]
    return collaborative_jobs

# Hybrid recommendation combining content and collaborative recommendations
def hybrid_recommendation(resume_skills):
    content_recommendations = content_based_recommendation(resume_skills)
    collaborative_recommendations = collaborative_recommendation(resume_skills)
    hybrid_jobs = {job["Position"]: score for job, score in content_recommendations}
    for _, job in collaborative_recommendations.iterrows():
        if job["Position"] not in hybrid_jobs:
            hybrid_jobs[job["Position"]] = 0.5  # Assign lower weight to collaborative recommendations
    sorted_jobs = sorted(hybrid_jobs.items(), key=lambda x: x[1], reverse=True)[:10]
    return [job for job, _ in sorted_jobs]

@app.route("/submit", methods=["POST"])
def submit_data():
    if "userfile" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["userfile"]
    filepath = save_file(file)
    try:
        data = ResumeParser(filepath).get_extracted_data()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    resume_skills = data.get("skills", [])
    resume_experience = data.get("experience", [])
    is_fresher = check_fresher(resume_experience)

    # Apply hybrid recommendation based on user profile
    job_recommendations = hybrid_recommendation(resume_skills)

    return render_template("page.html", job_list=job_recommendations, is_fresher=is_fresher)

if __name__ == "__main__":
    app.run(debug=True)
