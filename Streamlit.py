# from PIL import Image 
import pandas as pd
import numpy as np

import streamlit as st
import pickle
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the trained model
with open('C:\\Users\\shiva\\Downloads\\Final\\finalized_model.pkl', 'rb') as file:
    model, tfidf_vectorizer = pickle.load(file)
    
# Function to read text from PDF file
def read_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page_text = reader.pages[page_num].extract_text()
        try:
            text += page_text
        except UnicodeEncodeError as e:
            problematic_char = page_text[e.start]
            print(f"UnicodeEncodeError: Problematic character: {problematic_char} (Unicode: {ord(problematic_char)})")
            text += page_text.encode('utf-8').decode('utf-8', 'ignore')

    return text

# # Load English tokenizer, tagger, parser, NER, and word vectors
# import spacy
# nlp = spacy.load("en_core_web_sm")
# # Function to extract experience from text
# def extract_experience(text):
#     doc = nlp(text)
#     experience_years = []
#     skills = []
#     for ent in doc.ents:
#         if ent.label_ == "DATE":
#             # Check if the entity represents a year
#             if re.match(r'\b(19|20)\d{2}\b', ent.text):
#                 experience_years.append(ent.text)
#         elif ent.label_ == "ORG":
#             # Assuming organizations represent skills, you may need further validation
#             skills.append(ent.text)
#     return experience_years, skills


# Define the Streamlit app
def main():
    st.title("Resume Classification App")
    st.write("Upload a docx or PDF file to classify its type")

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["docx", "pdf"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        # Read the uploaded .docx file
        if file_extension == "docx":
                # Read the uploaded .docx file
            resume_text = docx2txt.process(uploaded_file)
        elif file_extension == "pdf":
            # Read the uploaded PDF file
            resume_text = read_pdf(uploaded_file)
            
        text= resume_text
        text = text.lower()										# Convert text to lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text)					# Remove special characters, numbers, and punctuation
        words = word_tokenize(text)								# Tokenize the text into words
        stop_words = set(stopwords.words('english'))			# Remove stop words
        words = [word for word in words if word not in stop_words]
        lemmatizer = WordNetLemmatizer()						# Lemmatize words to their base form
        words = [lemmatizer.lemmatize(word) for word in words]
        preprocessed_text = ' '.join(words)						# Join the words back into a single string
        # Preprocess text data:
        with open('resume_text.txt', 'w') as text_file:
            text_file.write(resume_text)
            
        with open('preprocessed_text.txt', 'w') as text_file:
            text_file.write(preprocessed_text)
      
        # Perform TF-IDF vectorization
        tfidf_vector = tfidf_vectorizer.transform([preprocessed_text])
        print("TF-IDF Vector Shape:", tfidf_vector.shape)
        
        # Convert sparse matrix to dense array
        tfidf_vector_dense = tfidf_vector.toarray()
        
        # Convert TF-IDF vector to DataFrame
        df_tfidf = pd.DataFrame(tfidf_vector_dense, columns=tfidf_vectorizer.get_feature_names_out())
        
        # Save TF-IDF vector to an Excel file
        df_tfidf.to_excel('tfidf_vector.xlsx', index=False)
        
        # Perform classification
        prediction = model.predict(tfidf_vector)
     

        # Display the prediction
        st.write("Prediction:", prediction)
        
        # # Extract experience
        # experience_years, skills = extract_experience(resume_text)
        # st.write("Experience years:", experience_years)
        # st.write("skills:", skills)

if __name__ == "__main__":
    main()
