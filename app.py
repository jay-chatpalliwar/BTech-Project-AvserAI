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
from flask import Flask, render_template, redirect, request, jsonify
from pyresparser import ResumeParser
import os
import time

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

# Define terms that indicate senior roles
senior_keywords = [
    "senior",
    "staff",
    "lead",
    "manager",
    "director",
    "head",
    "principal",
    "specialist",
    "sr.",
    "expert",
]


# Helper function to save files
def save_file(file):
    try:
        upload_folder = "uploads/"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Generate a unique filename
        filename = file.filename
        unique_filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(upload_folder, unique_filename)

        # Save the file
        file.save(filepath)
        return filepath
    except PermissionError:
        raise PermissionError("Permission denied when saving the file.")
    except Exception as e:
        raise Exception(f"An error occurred while saving the file: {str(e)}")


# Helper function to extract text from a PDF
def extract_text_from_pdf(filepath):
    try:
        text = ""
        reader = PdfReader(filepath)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


@app.route("/")
def hello():
    return render_template("page.html")


@app.route("/home")
def home():
    return redirect("/")


# def check_fresher(experience):
#     for exp in experience:
#         # Look for the word 'intern' anywhere in the description, case insensitive
#         if isinstance(exp, str):
#             if "intern" not in exp.lower():
#                 return False  # Return False if any non-intern experience is found
#     return True  # All experiences are internship-related


def check_fresher(experience):
    """
    Checks if a candidate is a fresher based on their experience.
    Returns True if all experience entries are internships or if only internship
    experience is present.

    Args:
        experience (list): List of experience-related strings
    Returns:
        bool: True if fresher (only internship experience), False otherwise
    """
    if not experience:
        return True

    # Find job title entries (typically the first two entries before dates)
    job_entries = []
    for exp in experience[
        :2
    ]:  # Look at first two entries which typically contain company and role
        if (
            isinstance(exp, str)
            and not exp.startswith("•")
            and not any(
                month in exp.lower()
                for month in [
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                ]
            )
        ):
            job_entries.append(exp.lower())

    # Check if any job entry contains 'intern'
    has_intern = any("intern" in entry for entry in job_entries)

    # If we found an internship and no other types of roles, return True
    return has_intern


# Example usage for the fresher check
# Output fresher status


@app.route("/submit", methods=["POST"])
def submit_data():
    try:
        if request.method == "POST":
            # Handle file upload
            if "userfile" not in request.files:
                return jsonify({"error": "No file part"}), 400

            file = request.files["userfile"]

            if file.filename == "":
                return jsonify({"error": "No selected file"}), 400

            # Save the uploaded file
            filepath = save_file(file)
            print("File saved successfully:", filepath)

            # Extract text from the uploaded PDF
            try:
                text = extract_text_from_pdf(filepath)
                print("Document opened successfully")
            except Exception as e:
                return jsonify({"error": str(e)}), 500

            # Parse resume data using ResumeParser
            try:
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                data = ResumeParser(filepath).get_extracted_data()
            except Exception as e:
                return jsonify({"error": f"Error parsing resume: {str(e)}"}), 500

            resume_skills = data.get("skills", [])
            resume_experience = data.get("experience", [])

            print("experience : ", resume_experience)

            # Check if the resume is from a fresher (no experience or only internship)
            # is_fresher = not resume_experience or all(
            #     "intern" in exp.lower() for exp in resume_experience if exp
            # )
            is_fresher = not resume_experience or check_fresher(resume_experience)

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

            # Filter jobs based on experience level
            if is_fresher:
                # Show jobs that are meant for freshers, excluding senior roles
                filtered_df = df[
                    ~df["Position"].str.contains("|".join(senior_keywords), case=False)
                ]
            else:
                # Show all job roles for experienced candidates
                filtered_df = df

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
    except PermissionError as pe:
        return jsonify({"error": str(pe)}), 500
    except FileNotFoundError as fnfe:
        return jsonify({"error": str(fnfe)}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
