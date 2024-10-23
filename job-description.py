from transformers import pipeline

# Load the summarization pipeline using BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# Function to summarize job description
def summarize_job_description_bart(text):
    # Abstractive summarization using BART
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]["summary_text"]


# Sample job description for testing
job_description = """
    We are looking for a highly skilled software engineer who is comfortable with both front and back end programming. 
    Full Stack Developers are responsible for developing and designing front end web architecture, ensuring the responsiveness of applications, and working alongside graphic designers for web design features, among other duties. 
    Full Stack Developers will be required to see out a project from conception to final product, requiring good organizational skills and attention to detail.
    Responsibilities: Developing front end website architecture. Designing user interactions on web pages. Developing back end website applications. 
    Creating servers and databases for functionality. Ensuring cross-platform optimization for mobile phones. Ensuring responsiveness of applications.
    Requirements: Degree in Computer Science. Strong organizational and project management skills. Proficiency in fundamental front-end languages such as HTML, CSS, and JavaScript. Familiarity with JavaScript frameworks such as Angular JS, React, and Amber. Proficiency in server-side languages such as Python, Ruby, Java, PHP, and .Net.
"""

# Generate summary
summary = summarize_job_description_bart(job_description)

print("Job Description Summary:")
print(summary)