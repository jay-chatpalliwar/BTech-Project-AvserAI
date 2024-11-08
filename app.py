"""
from pyresparser import ResumeParser
from flask import Flask, render_template, redirect, request
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

nltk.data.path.append("C:\\Users\\ACER/nltk_data")
nltk.download("stopwords")

stopw = set(stopwords.words("english"))
df = pd.read_csv("data.csv")
df["test"] = df["Job_Description"].apply(
    lambda x: " ".join(
        [word for word in str(x).split() if len(word) > 2 and word not in stopw]
    )
)

app = Flask(__name__)


# Helper functions
def is_fresher_job(job_description):
    fresher_keywords = [
        "fresher",
        "entry level",
        "junior",
        "0-1 years",
        "no experience",
    ]
    return any(
        keyword.lower() in job_description.lower() for keyword in fresher_keywords
    )


@app.route("/")
def hello():
    return render_template("page.html")


@app.route("/home")
def home():
    return redirect("/")


@app.route("/submit", methods=["POST"])
def submit_data():
    if request.method == "POST":
        f = request.files["userfile"]
        f.save(f.filename)
        print("Saved file:", f.filename)

        # Extract text from PDF and parse resume data
        text = ""
        try:
            reader = PdfReader(f.filename)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            print("Document opened successfully")

            # Load SpaCy model
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            data = ResumeParser(f.filename).get_extracted_data()
        except Exception as e:
            print("Error opening document:", e)
            data = ResumeParser(f.filename).get_extracted_data()

        # Extract skills from the resume
        resume_skills = data.get("skills", [])
        experience = data.get("total_experience", 0)  # Defaults to 0 if missing
        resume_experience_level = "fresher" if not experience else "experienced"

        # If resume has no skills, handle it gracefully
        if not resume_skills:
            resume_skills = [""]  # Placeholder if no skills found

        print("Extracted skills:", resume_skills)

        # Preprocess and vectorize skills for matching
        skills = []
        skills.append(" ".join(word for word in resume_skills))
        org_name_clean = skills

        def ngrams(string, n=3):
            string = fix_text(string)
            string = string.encode("ascii", errors="ignore").decode()
            string = string.lower()
            chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
            rx = "[" + re.escape("".join(chars_to_remove)) + "]"
            string = re.sub(rx, "", string)
            string = string.replace("&", "and")
            string = string.replace(",", " ")
            string = string.replace("-", " ")
            string = string.title()
            string = re.sub(" +", " ", string).strip()
            string = " " + string + " "
            string = re.sub(r"[,-./]|\sBD", r"", string)
            ngrams = zip(*[string[i:] for i in range(n)])
            return ["".join(ngram) for ngram in ngrams]

        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
        tfidf = vectorizer.fit_transform(org_name_clean)
        print("Vectorizing completed...")

        def getNearestN(query):
            queryTFIDF_ = vectorizer.transform(query)
            distances, indices = nbrs.kneighbors(queryTFIDF_)
            return distances, indices

        # Filter jobs based on experience level
        if resume_experience_level == "fresher":
            df_filtered = df[df["Job_Description"].apply(is_fresher_job)]
        else:
            df_filtered = df

        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
        unique_org = df_filtered["test"].values
        distances, indices = getNearestN(unique_org)
        unique_org = list(unique_org)
        matches = []

        for i, j in enumerate(indices):
            dist = round(distances[i][0], 2)
            temp = [dist]
            matches.append(temp)

        matches = pd.DataFrame(matches, columns=["Match confidence"])
        df_filtered["match"] = matches["Match confidence"]
        df1 = df_filtered.sort_values("match")
        df2 = (
            df1[["Position", "Company", "Location", "url", "test"]]
            .head(9)
            .reset_index()
        )

        df2["Location"] = df2["Location"].str.replace(r"[^\x00-\x7F]", "")
        df2["Location"] = df2["Location"].str.replace("â€“", "")
        dropdown_locations = sorted(df2["Location"].unique())

        # Generate relevance scores
        job_list = []
        relevance_scores = []

        for index, row in df2.iterrows():
            job_description = row["test"]
            resume_tfidf = vectorizer.transform(org_name_clean)
            job_tfidf = vectorizer.transform([job_description])

            relevance_score = cosine_similarity(resume_tfidf, job_tfidf)[0][0]
            relevance_scores.append(relevance_score)
            print(
                f"Relevance Score for {row['Position']} at {row['Company']}: {relevance_score}"
            )

            job_list.append(
                {
                    "Position": row["Position"],
                    "Company": row["Company"],
                    "Location": row["Location"],
                    "Apply Link": row["url"],
                }
            )

        print("\nTop Recommendations and Their Relevance Scores:")
        for i, row in enumerate(df2.itertuples()):
            print(
                f"{i+1}. Position: {row.Position}, Company: {row.Company}, Relevance Score: {relevance_scores[i]}"
            )

        return render_template(
            "page.html", job_list=job_list, dropdown_locations=dropdown_locations
        )


if __name__ == "__main__":
    app.run()
"""

from pyresparser import ResumeParser
from docx import Document
from flask import Flask, render_template, redirect, request
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

nltk.data.path.append("C:\\Users\\ACER/nltk_data")
nltk.download("stopwords")

stopw = set(stopwords.words("english"))
df = pd.read_csv("data.csv")
df["test"] = df["Job_Description"].apply(
    lambda x: " ".join(
        [word for word in str(x).split() if len(word) > 2 and word not in stopw]
    )
)

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("page.html")


@app.route("/home")
def home():
    return redirect("/")


@app.route("/submit", methods=["POST"])
def submit_data():
    if request.method == "POST":
        f = request.files["userfile"]
        f.save(f.filename)
        print("Saved file:", f.filename)

        text = ""
        try:
            reader = PdfReader(f.filename)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            print("Document opened successfully")

            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            data = ResumeParser(f.filename).get_extracted_data()
        except Exception as e:
            print("Error opening document:", e)
            data = ResumeParser(f.filename).get_extracted_data()

        resume_skills = data.get("skills", [])
        resume_experience = data.get("experience", [])

        is_fresher = not resume_experience or all(
            "intern" in exp.lower() for exp in resume_experience
        )

        print("Resume Skills:", resume_skills)
        print("Fresher status:", is_fresher)

        skills = [" ".join(resume_skills)]
        org_name_clean = skills

        def ngrams(string, n=3):
            string = fix_text(string)
            string = string.encode("ascii", errors="ignore").decode()
            string = string.lower()
            chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
            rx = "[" + re.escape("".join(chars_to_remove)) + "]"
            string = re.sub(rx, "", string)
            string = string.replace("&", "and")
            string = string.replace(",", " ")
            string = string.replace("-", " ")
            string = string.title()
            string = re.sub(" +", " ", string).strip()
            string = " " + string + " "
            string = re.sub(r"[,-./]|\sBD", r"", string)
            ngrams = zip(*[string[i:] for i in range(n)])
            return ["".join(ngram) for ngram in ngrams]

        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
        tfidf = vectorizer.fit_transform(org_name_clean)
        print("Vectorizing completed...")

        def getNearestN(query):
            queryTFIDF_ = vectorizer.transform(query)
            distances, indices = nbrs.kneighbors(queryTFIDF_)
            return distances, indices

        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
        unique_org = df["test"].values
        distances, indices = getNearestN(unique_org)
        unique_org = list(unique_org)
        matches = []

        for i, j in enumerate(indices):
            dist = round(distances[i][0], 2)
            temp = [dist]
            matches.append(temp)

        matches = pd.DataFrame(matches, columns=["Match confidence"])
        df["match"] = matches["Match confidence"]

        filtered_df = (
            df[df["Position"].str.contains("Intern|Junior", case=False)]
            if is_fresher
            else df
        )
        df_sorted = filtered_df.sort_values("match").head(9).reset_index()

        job_list = []
        relevance_scores = []

        for _, row in df_sorted.iterrows():
            job_description = row["test"]
            resume_tfidf = vectorizer.transform(org_name_clean)
            job_tfidf = vectorizer.transform([job_description])
            relevance_score = cosine_similarity(resume_tfidf, job_tfidf)[0][0]
            relevance_scores.append(relevance_score)
            job_list.append(
                {
                    "Position": row["Position"],
                    "Company": row["Company"],
                    "Location": row["Location"],
                    "Apply Link": row["url"],
                }
            )
            print(
                f"Relevance Score for {row['Position']} at {row['Company']}: {relevance_score}"
            )

        print("\nTop Recommendations and Their Relevance Scores:")
        for i, row in enumerate(df_sorted.itertuples()):
            print(
                f"{i+1}. Position: {row.Position}, Company: {row.Company}, Relevance Score: {relevance_scores[i]}"
            )

        dropdown_locations = sorted(df_sorted["Location"].unique())

        return render_template(
            "page.html", job_list=job_list, dropdown_locations=dropdown_locations
        )


if __name__ == "__main__":
    app.run()
