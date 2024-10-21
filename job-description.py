import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


def preprocess_text(text):
    # Remove any HTML tags
    text = re.sub("<[^<]+?>", "", text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    return sentences


def classify_sentence(sentence):
    # Keywords that might indicate responsibilities
    resp_keywords = set(
        [
            "develop",
            "manage",
            "create",
            "implement",
            "design",
            "coordinate",
            "lead",
            "analyze",
            "report",
            "maintain",
            "support",
            "collaborate",
            "ensure",
            "improve",
            "optimize",
        ]
    )

    # Keywords that might indicate requirements
    req_keywords = set(
        [
            "require",
            "must",
            "should",
            "need",
            "proficiency",
            "experience",
            "degree",
            "qualification",
            "skill",
            "knowledge",
            "ability",
            "familiar",
            "understanding",
        ]
    )

    # Preprocess the sentence
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(sentence.lower())
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stopwords.words("english")
    ]

    # Check for keyword matches
    resp_count = sum(1 for word in words if word in resp_keywords)
    req_count = sum(1 for word in words if word in req_keywords)

    # Classify based on keyword counts
    if resp_count > req_count:
        return "responsibility"
    elif req_count > resp_count:
        return "requirement"
    else:
        return "other"


def infer_responsibilities_requirements(job_description):
    """Infer responsibilities and requirements from the job description if not explicitly stated."""
    responsibilities = []
    requirements = []

    # Implicit responsibility indicators
    implicit_responsibilities = [
        "work closely",
        "oversee",
        "engage in",
        "support",
        "assist",
        "help",
        "ensure",
        "manage",
        "collaborate",
        "contribute",
        "focus on",
    ]

    # Implicit requirement indicators
    implicit_requirements = [
        "must have",
        "should be",
        "experience in",
        "knowledge of",
        "ability to",
        "familiarity with",
        "solid understanding",
        "strong understanding",
    ]

    for sentence in sent_tokenize(job_description):
        if any(phrase in sentence for phrase in implicit_responsibilities):
            responsibilities.append("Implicitly inferred: " + sentence)
        if any(phrase in sentence for phrase in implicit_requirements):
            requirements.append("Implicitly inferred: " + sentence)

    return responsibilities, requirements


def extract_sections(job_description):
    sentences = preprocess_text(job_description)

    responsibilities = []
    requirements = []

    for sentence in sentences:
        category = classify_sentence(sentence)
        if category == "responsibility":
            responsibilities.append(sentence)
        elif category == "requirement":
            requirements.append(sentence)

    # Infer responsibilities and requirements if not explicitly mentioned
    inferred_resp, inferred_req = infer_responsibilities_requirements(job_description)

    responsibilities.extend(inferred_resp)
    requirements.extend(inferred_req)

    return {"responsibilities": responsibilities, "requirements": requirements}


# Job descriptions with implied responsibilities and requirements
job_descriptions = [
    """
    Job Title: Software Engineer
    Location: Los Angeles, CA
    CTC: $100,000 - $120,000 per year
    Company: Tech Innovations Inc.

    We are seeking a Software Engineer to join our team. The ideal candidate will work closely with cross-functional teams to develop innovative software solutions. 

    The successful candidate will engage in code reviews, collaboration with product managers, and ensuring best practices in coding are followed. 

    The candidate must have a solid understanding of software development methodologies and be adaptable to new technologies.
    """,
    """
    Job Title: Project Manager
    Location: Remote
    CTC: $90,000 - $110,000 per year
    Company: Global Solutions

    We are looking for a Project Manager to oversee project execution and ensure timely delivery. This position requires strong leadership and communication skills.

    The ideal candidate should be able to work independently and manage multiple projects at the same time. 

    Familiarity with project management software and methodologies will be essential to ensure successful project delivery.
    """,
    """
    Job Title: Data Analyst
    Location: New York, NY
    CTC: $80,000 - $95,000 per year
    Company: Analytics Co.

    Analytics Co. is seeking a Data Analyst to analyze and interpret complex data sets.

    The candidate will use various data analysis tools and methodologies to draw insights from data, which will aid in making informed business decisions.

    Experience with data visualization and reporting tools will be crucial to effectively communicate findings to stakeholders.
    """,
    """
    Job Title: UX/UI Designer
    Location: San Francisco, CA
    CTC: $85,000 - $100,000 per year
    Company: Design Hub

    We are looking for a creative UX/UI Designer to create user-friendly interfaces.

    The ideal candidate will need to understand user needs and translate them into effective design solutions.

    The candidate should also have a strong understanding of design principles and tools such as Figma or Adobe XD.
    """,
    """
    Job Title: Marketing Coordinator
    Location: Boston, MA
    CTC: $60,000 - $75,000 per year
    Company: Market Creators

    The Marketing Coordinator will assist with campaign development and execution, working closely with the marketing team to enhance our brand visibility.

    The role requires a proactive attitude, and the ability to research market trends will be beneficial.

    Understanding digital marketing strategies and tools will help in driving effective marketing campaigns.
    """,
    """
    Job Title: DevOps Engineer
    Location: Seattle, WA
    CTC: $120,000 - $140,000 per year
    Company: CloudTech

    CloudTech is looking for a skilled DevOps Engineer to join our team.

    The candidate will work on improving deployment pipelines and automating processes to enhance operational efficiency.

    Strong problem-solving skills and experience with cloud services like AWS or Azure will be essential for this role.
    """,
    """
    Job Title: HR Generalist
    Location: Austin, TX
    CTC: $70,000 - $85,000 per year
    Company: People First

    The HR Generalist will support all human resource functions within the company.

    The candidate should be well-versed in recruitment, onboarding, and employee relations.

    Knowledge of employment laws and best practices in HR will be essential for success in this role.
    """,
    """
    Job Title: Frontend Developer
    Location: Chicago, IL
    CTC: $95,000 - $115,000 per year
    Company: Innovative Tech

    We are seeking a Frontend Developer to enhance our web applications.

    The ideal candidate will need to have a strong understanding of web development principles and design patterns.

    Familiarity with responsive design and experience with frameworks such as React will be crucial for this position.
    """,
    """
    Job Title: Backend Developer
    Location: Miami, FL
    CTC: $100,000 - $130,000 per year
    Company: Backend Masters

    Backend Masters is looking for a Backend Developer to build robust applications.

    The candidate will be responsible for optimizing server-side logic and database interactions to ensure high performance and responsiveness.

    Experience with technologies like Node.js, MongoDB, and RESTful APIs will be beneficial.
    """,
    """
    Job Title: Software Tester
    Location: Denver, CO
    CTC: $75,000 - $90,000 per year
    Company: Quality Checkers

    We are looking for a Software Tester to ensure the quality of our applications.

    The successful candidate should be familiar with testing methodologies and be able to design test cases.

    Experience with automation testing tools and a keen eye for detail will enhance the testing process.
    """,
]

# Iterate over each job description and extract sections
for job_description in job_descriptions:
    result = extract_sections(job_description)

    print("Responsibilities:")
    for item in result["responsibilities"]:
        print("- " + item)

    print("\nRequirements:")
    for item in result["requirements"]:
        print("- " + item)

    print("\n--------------------------------------------------\n")