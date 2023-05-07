#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# All imports
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

