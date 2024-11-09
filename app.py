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
df['test'] = df['Job_Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word) > 2 and word not in (stopw)]))
print(df["Location"])

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("home.html")

@app.route('/model')
def model():
    return render_template("model.html")

@app.route('/signin')
def signin():
    return render_template("signin.html")

@app.route('/signup')
def signup():
    return render_template("signup.html")

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
            
        resume = data['skills']
        print(resume)
    
        skills = []
        skills.append(' '.join(word for word in resume))
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
        
        def getNearestN(query):
            queryTFIDF_ = vectorizer.transform(query)
            distances, indices = nbrs.kneighbors(queryTFIDF_)
            return distances, indices

        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
        unique_org = (df['test'].values)
        distances, indices = getNearestN(unique_org)
        unique_org = list(unique_org)
        matches = []

        for i, j in enumerate(indices):
            dist = round(distances[i][0], 2)
            temp = [dist]
            matches.append(temp)

        matches = pd.DataFrame(matches, columns=['Match confidence'])
        df['match'] = matches['Match confidence']
        df1 = df.sort_values('match')
        df2 = df1[['Position', 'Company', 'Location', 'url']].head(9).reset_index()
        df2['Location'] = df2['Location'].str.replace(r'[^\x00-\x7F]', '')
        df2['Location'] = df2['Location'].str.replace("â€“", "")

        dropdown_locations = sorted(df2['Location'].unique())
        job_list = []
        
        for index, row in df2.iterrows():
            job_list.append({
                'Position': row['Position'],
                'Company': row['Company'],
                'Location': row['Location'],
                'Apply Link': row['url']
            })
        
        return render_template('page.html', job_list=job_list, dropdown_locations=dropdown_locations)

if __name__ == "__main__":
    app.run()
