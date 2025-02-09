import streamlit as st
import nltk
import re
import numpy as np
import torch
import pandas as pd
import plotly.express as px
from datetime import datetime

# Additional libraries for improved file parsing
import PyPDF2
import docx
from rake_nltk import Rake
import spacy
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from transformers import pipeline

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from sklearn.ensemble import IsolationForest

# ======================
# NLTK Downloads
# ======================
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ======================
# Load Models
# ======================
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# AI-generated content detection (DeBERTa-based)
ai_model_name = "microsoft/deberta-v3-base"
ai_tokenizer = AutoTokenizer.from_pretrained(ai_model_name)
ai_model = AutoModelForSequenceClassification.from_pretrained(ai_model_name)

# Generative model for explanations (DistilGPT2)
gen_model_name = "distilgpt2"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)

# Sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ======================
# Helper Functions
# ======================
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def parse_pdf(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    extracted_text = [page.extract_text() for page in pdf_reader.pages]
    return "\n".join(extracted_text)

def parse_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_resume(file):
    file_type = file.name.split(".")[-1].lower()
    try:
        if file_type == "pdf":
            return parse_pdf(file)
        elif file_type == "docx":
            return parse_docx(file)
        else:
            return file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading file: {e}"

def calculate_similarity(text1: str, text2: str) -> float:
    vec1 = similarity_model.encode(text1, convert_to_numpy=True)
    vec2 = similarity_model.encode(text2, convert_to_numpy=True)
    score = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return round(score * 100, 2)

def detect_ai_generated_text(text: str):
    inputs = ai_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = ai_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_score = probs[0][1].item()
    return ("AI-Generated" if ai_score > 0.5 else "Human-Written", round(ai_score * 100, 2))

def analyze_resume_tone(resume_text: str):
    results = sentiment_analyzer(resume_text)
    return results[0]['label'], results[0]['score']

def extract_keywords(text: str):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

def extract_entities(text: str):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def summarize_resume(resume_text: str):
    summary = summarizer(resume_text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def check_date_inconsistencies(resume: str):
    date_pattern = r"\b(20[0-2][0-9]|19[0-9]{2})\b"
    found_dates = re.findall(date_pattern, resume)
    if len(found_dates) > 1 and any(int(found_dates[i]) > int(found_dates[i + 1]) for i in range(len(found_dates) - 1)):
        return "⚠️ Inconsistent dates found."
    return "✅ No date inconsistencies."

def create_interactive_chart(data):
    fig = px.bar(data, x='Metric', y='Score', title='Resume Insights')
    return fig

# ====================
# Resume Analysis
# ====================
def analyze_resume(resume_text: str, job_description: str):
    preproc_resume = preprocess_text(resume_text)
    preproc_jd = preprocess_text(job_description)
    
    overall_match_score = calculate_similarity(preproc_jd, preproc_resume)
    
    weighted_score = min(
        (overall_match_score * 0.5) +
        (calculate_similarity("education qualifications required", preproc_resume) * 0.2) +
        (calculate_similarity("work experience required", preproc_resume) * 0.4) +
        (calculate_similarity("technical skills required", preproc_resume) * 0.3) +
        (calculate_similarity("certifications required", preproc_resume) * 0.1), 100)

    final_decision = "Strong Fit" if weighted_score > 70 else "Moderate Fit" if weighted_score > 50 else "Needs Improvement"

    ai_label, ai_confidence = detect_ai_generated_text(resume_text)
    resume_tone, tone_confidence = analyze_resume_tone(resume_text)
    
    extracted_keywords = extract_keywords(resume_text)
    extracted_entities = extract_entities(resume_text)
    resume_summary = summarize_resume(resume_text)
    date_check = check_date_inconsistencies(resume_text)
    
    return {
        "overall_match_score": overall_match_score,
        "weighted_score": weighted_score,
        "final_decision": final_decision,
        "ai_detection": {"label": ai_label, "confidence": ai_confidence},
        "resume_tone": {"label": resume_tone, "confidence": tone_confidence},
        "keywords": extracted_keywords,
        "entities": extracted_entities,
        "summary": resume_summary,
        "date_check": date_check
    }

# ====================
# Streamlit Interface
# ====================
def main():
    st.title("AI-Powered Resume Screening Tool")
    st.write("Upload a resume and compare it against a job description.")

    uploaded_resume = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    job_description = st.text_area("Paste the Job Description Here")

    if uploaded_resume and job_description:
        st.write("Analyzing Resume... Please wait.")
        resume_text = parse_resume(uploaded_resume)

        with st.spinner("Processing..."):
            results = analyze_resume(resume_text, job_description)

        st.subheader("Results")
        st.write(f"**Match Score:** {results['overall_match_score']}%")
        st.write(f"**Weighted Score:** {results['weighted_score']}%")
        st.write(f"**Final Decision:** {results['final_decision']}")
        st.write(f"**AI Detection:** {results['ai_detection']['label']} ({results['ai_detection']['confidence']}%)")
        st.write(f"**Resume Tone:** {results['resume_tone']['label']} ({round(results['resume_tone']['confidence'] * 100, 2)}%)")
        st.write(f"**Date Check:** {results['date_check']}")
        st.write(f"**Keywords:** {', '.join(results['keywords'][:10])}")
        st.write(f"**Named Entities:** {results['entities']}")
        st.write(f"**Summary:** {results['summary']}")

if __name__ == "__main__":
    main()
