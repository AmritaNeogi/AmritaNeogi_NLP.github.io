#!/usr/bin/env python
# coding: utf-8

# # Resume Screening using spaCy

# This project involves using spacy to perform entity recognition on 200 resumes and exploring different NLP tools for text analysis. The goal is to assist recruiters in quickly sifting through a large number of job applications. To aid hiring managers in determining whether to proceed to the interview stage, a skills match feature has been added, which uses a metric. Two datasets will be used, one containing resume texts and the other containing skills that will be used to create an entity ruler.

# # Imports

# In[1]:


#Data loading/ Data manipulation
import pandas as pd
import numpy as np

#spacy
import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc


 #nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


#visualize
import plotly.express as px
import matplotlib.pyplot as plt
from spacy import displacy



# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")


# # Loading Data

# In[2]:


df = pd.read_csv("Resume.csv")
df = df.reindex(np.random.permutation(df.index))
data = df.copy().iloc[
    0:200,
]
data.head()


# # Loading spaCy Model

# In[3]:


nlp = spacy.load("en_core_web_lg")
skill_pattern_path = "jz_skill_patterns.jsonl"


# # Entity Ruler

# In[4]:


ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_pattern_path)
nlp.pipe_names


# # Skiils

# In[5]:


def get_skills(text):
    doc = nlp(text)
    myset = []
    subset = []
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            subset.append(ent.text)
    myset.append(subset)
    return subset


def unique_skills(x):
    return list(set(x))


# # Resume Text Cleaning

# In[6]:


nltk.download(['stopwords','wordnet'])
nltk.download('omw-1.4')


# In[7]:


clean = []
for i in range(data.shape[0]):
    review = re.sub(
        '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
        " ",
        data["Resume_str"].iloc[i],
    )
    review = review.lower()
    review = review.split()
    lm = WordNetLemmatizer()
    review = [
        lm.lemmatize(word)
        for word in review
        if not word in set(stopwords.words("english"))
    ]
    review = " ".join(review)
    clean.append(review)


# # Implementing the Functions

# In[8]:


data["Clean_Resume"] = clean
data["skills"] = data["Clean_Resume"].str.lower().apply(get_skills)
data["skills"] = data["skills"].apply(unique_skills)
data.head()


# In[10]:


# set the random seed
np.random.seed(42)

fig = px.histogram(data, x="Category", title="Distribution of Jobs Categories", color="Category",
                   color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_xaxes(categoryorder="total descending")

fig.show()


# In[11]:


Job_cat = data["Category"].unique()
Job_cat = np.append(Job_cat, "ALL")


# In[12]:


# We want to know the skills required for someone apply for a BUSINESS-DEVELOPMENT job

Job_Category= 'BUSINESS-DEVELOPMENT'

Total_skills = []
if Job_Category != "ALL":
    fltr = data[data["Category"] == Job_Category]["skills"]
    for x in fltr:
        for i in x:
            Total_skills.append(i)
else:
    fltr = data["skills"]
    for x in fltr:
        for i in x:
            Total_skills.append(i)

fig = px.histogram(data, x=Total_skills, title=f"{Job_Category} Distribution of Skills",
                   color=Total_skills,
                   color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_xaxes(categoryorder="total descending")
fig.show()


# # Entity Recognition

# In[14]:


sent = nlp(data["Resume_str"].iloc[0])
displacy.render(sent, style="ent", jupyter=True)


# # Dependency Parsing

# In[15]:


displacy.render(sent[0:10], style="dep", jupyter=True, options={"distance": 90})


# # Custom Entity Recognition

# In[16]:


patterns = df.Category.unique()
for a in patterns:
    ruler.add_patterns([{"label": "Job-Category", "pattern": a}])


# In[17]:


# options=[{"ents": "Job-Category", "colors": "#ff3232"},{"ents": "SKILL", "colors": "#56c426"}]
colors = {
    "Job-Category": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
    "SKILL": "linear-gradient(90deg, #9BE15D, #00E3AE)",
    "ORG": "#ffd966",
    "PERSON": "#e06666",
    "GPE": "#9fc5e8",
    "DATE": "#c27ba0",
    "ORDINAL": "#674ea7",
    "PRODUCT": "#f9cb9c",
}
options = {
    "ents": [
        "Job-Category",
        "SKILL",
        "ORG",
        "PERSON",
        "GPE",
        "DATE",
        "ORDINAL",
        "PRODUCT",
    ],
    "colors": colors,
}
sent = nlp(data["Resume_str"].iloc[5])
displacy.render(sent, style="ent", jupyter=True, options=options)



# # Resume Anlaysis

# In[18]:


# taking example of a random resume 
input_resume= "Jonathan Andrews Data Scientist I am a certified data scientist professional, who loves building machine learning models and blogs about the latest AI technologies. I am currently testing AI Products at PEC-PITC, which later gets approved for human trials. jandrews@email.com +112345678 Phoenix, Arizona abidaliawan.me WORK EXPERIENCE Data Scientist Arizona Innovation and Testing Center - PEC 04/2021 - Present, Phoenix, Arizona Redesigned data of engineers that were mostly scattered and unavailable. Designed dashboard and data analysis report to help higher management make better decisions. Accessibility of key information has created a new culture of making data-driven decisions. Contact: Rameez Ali - rameezali@email.com Data Scientist Freelancing/Kaggle 11/2020 - Present, Phoenix, Arizona Engineered a healthcare system. Used machine learning to detect some of the common decisions. The project has paved the way for others to use new techniques to get better results. Participated in Kaggle machine learning competitions. Learned new techniques to get a better score and finally got to 1 percent rank. Researcher / Event Organizer CREDIT 02/2017 - 07/2017, Kuala Lumpur, Malaysia Marketing for newly build research lab. Organized technical events and successfully invited the multiple company's CEO for talks. Reduced the gap between industries and educational institutes. Research on new development in the IoT sector. Created research proposal for funding. Investigated the new communication protocol for IoT devices. Contact: Dr. Tan Chye Cheah - dr.chyecheah.t@apu.edu.my EDUCATION MSc in Technology Management Staffordshire University 11/2015 - 04/2017, Postgraduate with Distinction Challenges in Implementing IoT-enabled Smart cities in Malaysia. Bachelors Electrical Telecommunication Engineering COMSATS Institute of Information Technology, Phoenix 08/2010 - 01/2014, CGPA: 4.00 Networking Satellite communications Programming/ Matlab Telecommunication Engineering SKILLS Designing Leadership Media/Marketing R/Python SQL Tableau NLP Data Analysis Machine learning Deep learning Webapp/Cloud Feature Engineering Ensembling Time Series Technology Management ACHIEVEMENTS 98th Hungry Geese Simulation Competition (08/2021) 2nd in Covid-19 vaccinations around the world (07/2021) 8th in Automatic Speech Recognition in WOLOF (06/2021) Top 10 in WiDS Datathon. (03/2021) 40th / 622 in MagNet: Model the Geomagnetic Field Hosted by NOAA (02/2021) 18th in Rock, Paper, Scissors/Designing AI Agent Competition. (02/2021) PROJECTS Goodreads Profile Analysis WebApp (09/2021) Data Analysis Web Scraping XLM Interactive Visualization Contributed in orchest.io (08/2021) Testing and Debuging Technical Article Proposing new was to Improve ML pipelines World Vaccine Update System (06/2021) Used sqlite3 for database Automated system for daily update the Kaggle DB and Analysis Interactive dashboard mRNA-Vaccine-Degradation-Prediction (06/2021) Explore our dataset and then preprocessed sequence, structure, and predicted loop type features Train deep learning GRU model Trip Advisor Data Analysis/ML (04/2021) Preprocessing Data, Exploratory Data analysis, Word clouds. Feature Engineering, Text processing. BiLSTM Model for predicting rating, evaluation, model performance. Jane Street Market Prediction (03/2021) EDA, Feature Engineering, experimenting with hyperparameters. Ensembling: Resnet, NN Embeddings, TF Simple NN model. Using simple MLP pytorch model. Achievements/Tasks Achievements/Tasks Achievements/Tasks Thesis Courses"


# In[19]:


sent2 = nlp(input_resume)
displacy.render(sent2, style="ent", jupyter=True, options=options)


# # Accuracy

# In[20]:


input_skills = "Database, SQL, Machine Learning" # example of input skills


# In[21]:


req_skills = input_skills.lower().split(",")
resume_skills = unique_skills(get_skills(input_resume.lower()))
score = 0
for x in req_skills:
    if x in resume_skills:
        score += 1
req_skills_len = len(req_skills)
match = round(score / req_skills_len * 100, 1)

print(f"The current Resume is {match}% matched to your requirements")


# We can also see the skills mentioned in the input resume.

# In[23]:


print(resume_skills)


# In[ ]:




