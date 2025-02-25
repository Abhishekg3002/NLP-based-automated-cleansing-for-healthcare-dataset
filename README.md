# NLP-based-automated-cleansing-for-healthcare-dataset
import pandas as pd
import numpy as np
import re
import nltk
import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from spellchecker import SpellChecker
from collections import Counter

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Download required NLTK packages
nltk.download('stopwords')
nltk.download('punkt')

# Initialize SpellChecker
spell = SpellChecker()

# Function to load healthcare data (assuming CSV file format)
def load_data(file_path):
    return pd.read_csv(file_path)

# Step 1: Basic text preprocessing
def clean_text(text):
    # Lowercase all text
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Step 2: Spelling correction
def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) for word in words]
    return ' '.join(corrected_words)

# Step 3: Remove stopwords
def remove_stopwords(text):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

# Step 4: Standardize terms and abbreviations
def standardize_terms(text):
    # Example dictionary of common abbreviations in healthcare
    terms_dict = {
        'bp': 'blood pressure',
        'hr': 'heart rate',
        'dx': 'diagnosis',
        'sx': 'symptoms',
        'meds': 'medications',
        'cns': 'central nervous system',
        'wbc': 'white blood cells'
    }
    
    for abbreviation, full_term in terms_dict.items():
        text = text.replace(abbreviation, full_term)
    
    return text

# Step 5: Named Entity Recognition (NER) to standardize medical entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Step 6: Duplicate removal
def remove_duplicates(df):
    # Remove duplicate rows based on 'patient_id' or any relevant column
    df.drop_duplicates(subset='patient_id', keep='last', inplace=True)
    return df

# Step 7: Entity Matching (Medical term matching)
def match_medical_terms(text, medical_terms):
    matched_terms = []
    for term in medical_terms:
        if fuzz.partial_ratio(text, term) > 80:  # Matching threshold
            matched_terms.append(term)
    return matched_terms

# Example function to standardize medical terms (could be a dictionary or external API)
def medical_standardization(text):
    medical_terms = ['hypertension', 'diabetes', 'stroke', 'cancer', 'asthma']
    matched_terms = match_medical_terms(text, medical_terms)
    return matched_terms

# Step 8: Full cleansing pipeline
def cleanse_healthcare_data(df):
    # Clean text columns in the dataframe
    for index, row in df.iterrows():
        text = row['clinical_notes']
        
        # Step 1: Basic text preprocessing
        text = clean_text(text)
        
        # Step 2: Spelling correction
        text = correct_spelling(text)
        
        # Step 3: Remove stopwords
        text = remove_stopwords(text)
        
        # Step 4: Standardize terms and abbreviations
        text = standardize_terms(text)
        
        # Step 5: Extract entities (optional)
        entities = extract_entities(text)
        
        # Step 6: Standardize medical terms
        matched_terms = medical_standardization(text)
        
        # Store the cleaned text back into the dataframe
        df.at[index, 'cleaned_notes'] = text
        df.at[index, 'extracted_entities'] = str(entities)
        df.at[index, 'matched_medical_terms'] = str(matched_terms)
    
    # Step 7: Remove duplicate rows (based on patient_id)
    df = remove_duplicates(df)
    
    return df

# Sample file path to healthcare data
file_path = 'healthcare_data.csv'

# Step 9: Execute the cleansing pipeline
df = load_data(file_path)
cleaned_df = cleanse_healthcare_data(df)

# Save the cleaned dataframe to a new CSV file
cleaned_df.to_csv('cleaned_healthcare_data.csv', index=False)

print("Healthcare data cleansing complete!")

