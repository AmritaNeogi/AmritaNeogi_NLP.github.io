# Dockerfile, Image, Container
FROM python:3.11.3

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/AmritaNeogi/AmritaNeogi.github.io.git

WORKDIR /AmritaNeogi.github.io/NER

RUN pip install pandas numpy spacy nltk plotly matplotlib 

CMD ["python", "Resume_Screening.py"]
