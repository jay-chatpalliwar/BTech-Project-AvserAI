from transformers import pipeline

# Load a pre-trained summarization model
summarizer = pipeline("summarization")

# Input job description text
job_description = """
Parakh is a B2B Edtech SaaS company based out of New Delhi.
We are a fast pace growing organization whose idea is to bring a fundamental change in the current process of activities taking place in the assessment & education industry. We are making them relevant to the digital age by offering industry specific technological solutions at affordable prices.
Main Duties and Responsibilities:
Translate application storyboards and use cases into functional applications.
Design, build, and maintain efficient, reusable, and reliable code.
Build and maintain customized applications and reports.
Integrate data from various backend services and databases.
Ensure the best possible performance, quality, and responsiveness of applications.
Identify bottlenecks and bugs, and devise solutions to mitigate and address these issues.
Help maintain code quality, organization, and automatization.
Proficient in Core Java, Spring Boot, SQL Server. Additional skills include familiarity with object-oriented programming, design patterns, and experience with Git and SVN.
"""

# Generate a summary with a pre-trained model
summary = summarizer(job_description, max_length=200, min_length=50, do_sample=False)

# Print the summary
print("Job Summary:", summary[0]['summary_text'])
