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
nltk.data.path.append('C:\\Users\\ACER/nltk_data')
nltk.download('stopwords')

stopw = set(stopwords.words('english'))

df = pd.read_csv('data.csv')
df['test'] = df['Job_Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word) > 2 and word not in stopw]))
print(df["Location"])
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("page.html")

@app.route("/home")
def home():
    return redirect('/')

@app.route('/submit', methods=['POST'])
def submit_data():
    if request.method == 'POST':
        f = request.files['userfile']
        f.save(f.filename)
        print("Saved file:", f.filename)

        text = ""
        try:
            reader = PdfReader(f.filename)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            print("Document opened successfully")

            # Load SpaCy model
            nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
            data = ResumeParser(f.filename).get_extracted_data()

        except Exception as e:
            print("Error opening document:", e)
            data = ResumeParser(f.filename).get_extracted_data()

        # Extract skills and experience
        resume_skills = data.get('skills', [])
        resume_experience_years = data.get('total_experience', 0)

        print(f"Skills: {resume_skills}")
        print(f"Experience: {resume_experience_years} years")

        # Process skills for vectorization
        skills = [' '.join(word for word in resume_skills)]
        org_name_clean = skills

        def ngrams(string, n=3):
            string = fix_text(string)
            string = string.encode("ascii", errors="ignore").decode()
            string = string.lower()
            chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
            rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
            string = re.sub(rx, '', string)
            string = string.replace('&', 'and')
            string = string.replace(',', ' ')
            string = string.replace('-', ' ')
            string = string.title()
            string = re.sub(' +', ' ', string).strip()
            string = ' ' + string + ' '
            string = re.sub(r'[,-./]|\sBD', r'', string)
            ngrams = zip(*[string[i:] for i in range(n)])
            return [''.join(ngram) for ngram in ngrams]

        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
        tfidf = vectorizer.fit_transform(org_name_clean)
        print('Vectorizing completed...')

        # Add experience-based matching logic
        def getNearestN(query_skills, query_experience):
            queryTFIDF_ = vectorizer.transform(query_skills)
            distances, indices = nbrs.kneighbors(queryTFIDF_, n_neighbors=10)

            required_experience = df['required_experience'].values
            experience_matches = []
            for i, index in enumerate(indices):
                job_exp = required_experience[index[0]]
                experience_diff = abs(query_experience - job_exp) 

                combined_score = distances[i][0] + (experience_diff * 0.1)
                experience_matches.append(combined_score)

            return experience_matches, indices

        # Nearest neighbors model based on skills
        nbrs = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(tfidf)
        unique_org = df['test'].values

        experience_matches, indices = getNearestN(unique_org, resume_experience_years)

        matches = []
        for i, j in enumerate(indices):
            dist = round(experience_matches[i], 2)
            matches.append([dist])

        matches = pd.DataFrame(matches, columns=['Match confidence'])
        df['match'] = matches['Match confidence']

        df1 = df.sort_values('match')
        df2 = df1[['Position', 'Company', 'Location', 'url', 'required_experience']].head(9).reset_index()
        df2['Location'] = df2['Location'].str.replace(r'[^\x00-\x7F]', '')
        df2['Location'] = df2['Location'].str.replace("â€“", "")

        dropdown_locations = sorted(df2['Location'].unique())
        job_list = []
        for index, row in df2.iterrows():
            job_list.append({
                'Position': row['Position'],
                'Company': row['Company'],
                'Location': row['Location'],
                'Required Experience': row['required_experience'],
                'Apply Link': row['url']
            })

        return render_template('page.html', job_list=job_list, dropdown_locations=dropdown_locations)

if __name__ == "__main__":
    app.run()
