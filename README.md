# perform-NER-for-resume-screening
This is a comprehensive tutorial on how to perform Named Entity Recognition (NER) for resume screening using spaCy.

For the tutorial, please refer to the [website](https://amritaneogi.github.io/)


Dataset

livecareer.com resume Dataset
A collection of 2400+ Resume Examples taken from livecareer.com for categorizing a given resume into any of the labels defined in the dataset: Resume [Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).

Inside the CSV

ID: Unique identifier and file name for the respective pdf.
Resume_str : Contains the resume text only in string format.
Resume_html : Contains the resume data in html format as present while web scrapping.
Category : Category of the job the resume was used to apply.
Present categories
HR, Designer, Information-Technology, Teacher, Advocate, Business-Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital-Media, Automobile, Chef, Finance, Apparel, Engineering, Accountant, Construction, Public-Relations, Banking, Arts, Aviation

Acknowledgements

Data was obtained by scrapping individual resume examples from www.livecareer.com website. 
Web Scrapping code is taken from [GitHub Repo](https://github.com/Sbhawal/resumeScraper).

