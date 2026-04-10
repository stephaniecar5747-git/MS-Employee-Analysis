import pandas as pd
import string
import os
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
import spacy

# Load spaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")

# Load raw data into a data frame
df_raw = pd.read_csv("C:/Users/HP/OneDrive/Desktop/CUCEA 2025 B/2do Semestre/Programacion II/proyecto_challenge/Basico/data/reviews_microsoft.csv")
df_raw.head()

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = "".join([i for i in text if not i.isdigit()]) # Remove numbers
    return text

def process_corpus(text, lang='en'):
    if lang == 'en':
        doc = nlp_en(text)
        stops = set(stopwords.words('english')) - {'not', 'no', 'won', 'dont', 'isnt'} # Removed words from stopwords list to avoid incorrect analysis
    else:
        doc = nlp_es(text)
        stops = set(stopwords.words('spanish')) - {'no', 'ni', 'poco', 'nada'} # Removed words from stopwords list to avoid incorrect analysis
    return " ".join([token.lemma_ for token in doc if token.text not in stops and not token.is_space])

# --- 1. DATA PREP ---
# Use your df_raw here
df = df_raw[['location','dates','job-title', 'summary','pros','cons', 'overall-ratings']].copy()
df['dates'] = pd.to_datetime(df['dates'], errors='coerce')
df = df.dropna().head(1000)

# --- 2. CLEAN PROS & CONS ---
for col in ['pros', 'cons']:
    df[f'{col}_clean'] = df[col].apply(clean_text)
    # Apply lemmatization to Pros and Cons specifically
    df[f'{col}_lemmas'] = df[f'{col}_clean'].apply(lambda x: process_corpus(x, lang='en'))

# Combine for the SVM model feature
df['full_review_lemmas'] = df['pros_lemmas'] + " " + df['cons_lemmas']

# --- 3. SAVE ---
if not os.path.exists('data'): os.makedirs('data')
df.to_csv("data/processed_en.csv", index=False)
print("Preprocessing: Pros and Cons lemmatized and saved.")