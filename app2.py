import streamlit as st

# ==============================
# 1) Set Page Config FIRST!
# ==============================
st.set_page_config(page_title="Resume Inspector (DeBERTa + Semantic Search)", layout="wide")

# =====================================================
# 2) Imports and Basic Setup
# =====================================================
import os
import warnings
import nltk
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import PyPDF2
import docx
import spacy
from difflib import SequenceMatcher
from dateutil import parser as date_parser
from sklearn.ensemble import IsolationForest

# For scraping LinkedIn (if needed)
import requests
from bs4 import BeautifulSoup

# Sentence Transformers for semantic search and plagiarism detection
from sentence_transformers import SentenceTransformer, util

import torch
import tensorflow as tf
from transformers import pipeline

warnings.filterwarnings("ignore")
nltk.download('punkt', quiet=True)
tf.compat.v1.reset_default_graph()

# =====================================================
# 3) SpaCy Model Loading with Fallback
# =====================================================
fallback_spacy_message = None
try:
    spacy_model = spacy.load("en_core_web_lg")
except OSError:
    spacy_model = spacy.load("en_core_web_sm")
    fallback_spacy_message = (
        "The 'en_core_web_lg' model is not installed. Falling back to 'en_core_web_sm'.\n"
        "Run `python -m spacy download en_core_web_lg` for the larger model."
    )

# =====================================================
# 4) Model Names and Loading
# =====================================================
DEBERTA_MODEL_NAME = "microsoft/deberta-v3-base"
LIGHTWEIGHT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PLAGIARISM_ST_MODEL = "all-mpnet-base-v2"

@st.cache_resource
def load_models():
    return {
        "zero_shot_classifier": pipeline("zero-shot-classification", model=DEBERTA_MODEL_NAME),
        "similarity_model": SentenceTransformer(PLAGIARISM_ST_MODEL),
        "semantic_search_model": SentenceTransformer(LIGHTWEIGHT_ST_MODEL),
        "text_generator": pipeline("text-generation", model="distilgpt2", device=-1, max_new_tokens=50),
        "ner_model": spacy_model,
    }

models = load_models()

# =====================================================
# 5) Visualization Configuration
# =====================================================
COLOR_SCALE = px.colors.sequential.Viridis
RISK_COLORS = {"High": "#FF474C", "Medium": "#FFA500", "Low": "#2ECC71"}

def calculate_overall_risk(risks_dict):
    numeric_vals = [v for v in risks_dict.values() if isinstance(v, (int, float))]
    return sum(numeric_vals) / len(numeric_vals) if numeric_vals else 0

def create_risk_assessment(risks):
    overall_risk = calculate_overall_risk(risks)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=overall_risk,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Risk Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': RISK_COLORS["Low"]},
                    {'range': [30, 70], 'color': RISK_COLORS["Medium"]},
                    {'range': [70, 100], 'color': RISK_COLORS["High"]}
                ]
            }
        )
    )
    return fig

def create_skill_radar(metrics):
    categories = list(metrics.keys())
    values = list(metrics.values())
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + values[:1],
            theta=categories + categories[:1],
            fill='toself'
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=400
    )
    return fig

# =====================================================
# 6) Custom Criteria (Skills, Education, Experience)
# =====================================================
custom_criteria = {
    "Data Scientist": {
        "required_skills": ["python", "machine learning", "data analysis", "statistics", "deep learning", "sql"],
        "education": ["bachelor", "master", "phd", "computer science", "mathematics"],
        "experience": ["data science", "research", "modeling", "analytics"]
    },
    "Software Engineer": {
        "required_skills": ["java", "python", "c++", "software development", "api", "cloud", "agile"],
        "education": ["bachelor", "computer science", "engineering"],
        "experience": ["software engineering", "devops", "backend", "frontend", "full stack"]
    },
    "Product Manager": {
        "required_skills": ["product roadmap", "agile", "stakeholder management", "market research", "user stories"],
        "education": ["mba", "business", "marketing", "management"],
        "experience": ["product management", "strategy", "go-to-market"]
    }
}

# =====================================================
# 7) Parsing and Basic Fraud Detection Functions
# =====================================================
def parse_resume(file) -> str:
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text() or "")
        return " ".join(text)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc_ = docx.Document(file)
        return " ".join([para.text for para in doc_.paragraphs])
    elif file.type == "text/plain":
        return file.read().decode("utf-8", errors="ignore")
    else:
        st.warning("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
        return ""

def detect_date_anomalies(text):
    date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})\b'
    dates = []
    for match in re.finditer(date_pattern, text):
        try:
            d = date_parser.parse(match.group(), fuzzy=True)
            dates.append(d)
        except:
            pass
    anomalies = []
    if len(dates) > 1:
        sorted_d = sorted(dates)
        for i in range(1, len(sorted_d)):
            if (sorted_d[i] - sorted_d[i-1]).days < 0:
                anomalies.append(f"Reversed/Overlapping Dates: {sorted_d[i-1]} -> {sorted_d[i]}")
    return dates, anomalies

def detect_contact_inconsistencies(text):
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phones = re.findall(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    return {
        "email_risk": "High" if len(emails) == 0 else "Low",
        "phone_risk": "High" if len(phones) == 0 else "Low",
        "email_domains": list(set(e.split('@')[-1] for e in emails)),
        "unique_phones": len(set(phones))
    }

def detect_copied_content(resume_text, jd_text):
    resume_sents = [s.strip() for s in nltk.sent_tokenize(resume_text)]
    jd_sents = [s.strip() for s in nltk.sent_tokenize(jd_text)]
    similarities = []
    copied_pairs = []
    for rs in resume_sents:
        for js in jd_sents:
            ratio = SequenceMatcher(None, rs, js).ratio()
            if ratio > 0.8:
                similarities.append(ratio)
                copied_pairs.append((rs, js, ratio))
    plag_score = (np.mean(similarities)*100) if similarities else 0
    return {
        "plagiarism_score": round(plag_score, 2),
        "copied_pairs": copied_pairs,
        "total_matches": len(copied_pairs)
    }

# =====================================================
# 8) Skill Analysis Functions
# =====================================================
def enhanced_skill_analysis(resume_text, jd_text, role):
    if role not in custom_criteria:
        return {"technical": 0, "domain": 0, "experience": 0, "education": 0}
    data = custom_criteria[role]
    skills = data["required_skills"]
    edu = data["education"]
    exp = data["experience"]
    r_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    tech_score = _compute_keyword_score(r_lower, skills)
    edu_score = _compute_keyword_score(r_lower, edu)
    exp_score = _compute_keyword_score(r_lower, exp)
    jd_words = set(jd_lower.split())
    resume_words = set(r_lower.split())
    overlap = len(jd_words.intersection(resume_words))
    domain_score = (overlap / (len(jd_words)+1)) * 100
    return {
        "technical": tech_score,
        "domain": round(domain_score, 2),
        "experience": exp_score,
        "education": edu_score
    }

def _compute_keyword_score(text, keywords):
    if not keywords:
        return 0
    matched = sum(1 for kw in keywords if kw.lower() in text)
    return round((matched / len(keywords)) * 100, 2)

# =====================================================
# 9) Compare Resume Responsibilities with Job Description
# =====================================================
def compare_resume_responsibilities(resume_text, jd_text, role):
    """
    Extracts flagged responsibilities from the resume and uses a zero-shot classification
    pipeline to compare each responsibility against the job description.
    Candidate labels:
      - "aligned with job description"
      - "not aligned with job description"
    Returns a list of dictionaries with:
      - responsibility
      - reasoning (confidence percentage)
      - HR Note (brief message)
    """
    flagged = flag_responsibilities_projects(resume_text)
    results = []
    if not flagged:
        return results
    classifier = models["zero_shot_classifier"]
    candidate_labels = ["aligned with job description", "not aligned with job description"]
    hypothesis_template = f"This responsibility is {{}} for the job description for a {role} role."
    for item in flagged:
        responsibility = item["excerpt"]
        pred = classifier(
            responsibility,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False
        )
        if not pred["labels"]:
            continue
        top_label = pred["labels"][0]
        score = pred["scores"][0] if pred["scores"] else 0.0
        reasoning = f"{score*100:.1f}%"
        hr_note = ("Responsibility appears to align with the job description."
                   if top_label == "aligned with job description"
                   else "Responsibility does not align with the job description.")
        results.append({
            "responsibility": responsibility,
            "reasoning": reasoning,
            "HR Note": hr_note
        })
    return results

# =====================================================
# 10) Generative Analysis of Flagged Responsibilities (with error handling)
# =====================================================
def generative_analysis_responsibility(responsibility_text, role):
    prompt = (
        f"Analyze the following responsibility from a candidate's resume for the role of {role}.\n"
        f"Responsibility: \"{responsibility_text}\"\n"
        "Answer in the following format:\n"
        "Verdict: [Yes/No]; Reasoning: [brief explanation]; Confidence: [number]%\n"
    )
    try:
        generated = models["text_generator"](prompt, max_new_tokens=50, num_return_sequences=1)
        output = generated[0]["generated_text"]
        parts = output.split(";")
        # Check that we have at least three parts
        if len(parts) < 3:
            raise ValueError("Not enough output parts.")
        verdict = parts[0].replace("Verdict:", "").strip()
        reasoning = parts[1].replace("Reasoning:", "").strip()
        confidence = parts[2].replace("Confidence:", "").strip()
        return {
            "responsibility": responsibility_text,
            "verdict": verdict,
            "reasoning": reasoning,
            "confidence": confidence
        }
    except Exception as e:
        return {
            "responsibility": responsibility_text,
            "verdict": "N/A",
            "reasoning": "Error occurred",
            "confidence": "N/A"
        }

def analyze_flagged_responsibilities(resume_text, role):
    flagged = flag_responsibilities_projects(resume_text)
    analysis_results = []
    for item in flagged:
        res_text = item["excerpt"]
        result = generative_analysis_responsibility(res_text, role)
        analysis_results.append(result)
    return analysis_results

# =====================================================
# 11) Detailed Strength & Weakness Analysis
# =====================================================
def generate_detailed_analysis(analysis_results, role):
    lines = []
    sm = analysis_results.get("skill_match", {})
    tech = sm.get("technical", 0)
    domain = sm.get("domain", 0)
    exp = sm.get("experience", 0)
    edu = sm.get("education", 0)
    if tech > 70:
        lines.append(f"Strength: Excellent technical skill coverage ({tech}%) for {role}.")
    elif tech < 40:
        lines.append(f"Weakness: Low technical skill match ({tech}%).")
    else:
        lines.append(f"Neutral: Moderate technical skill coverage ({tech}%).")
    if domain > 70:
        lines.append(f"Strength: Strong domain knowledge overlap ({domain}%) with the JD.")
    elif domain < 40:
        lines.append(f"Weakness: Low domain knowledge overlap ({domain}%).")
    else:
        lines.append(f"Neutral: Fair domain overlap ({domain}%).")
    if exp > 70:
        lines.append(f"Strength: High relevance in experience ({exp}%).")
    elif exp < 40:
        lines.append(f"Weakness: Low experience match ({exp}%).")
    else:
        lines.append(f"Neutral: Average experience relevance ({exp}%).")
    if edu > 70:
        lines.append(f"Strength: Education background strongly fits ({edu}%).")
    elif edu < 40:
        lines.append(f"Weakness: Education background may be lacking ({edu}%).")
    else:
        lines.append(f"Neutral: Moderate education fit ({edu}%).")
    pl_score = analysis_results.get("plagiarism_score", 0)
    if pl_score > 40:
        lines.append(f"Weakness: High overlap with JD content (Plagiarism ~ {pl_score}%).")
    elif pl_score == 0:
        lines.append("Strength: No overlap with JD content (0% plagiarism).")
    else:
        lines.append(f"Neutral: Some overlap with JD content (~{pl_score}%).")
    date_anoms = analysis_results.get("date_anomalies", [])
    if date_anoms:
        lines.append(f"Weakness: {len(date_anoms)} date anomalies found.")
    else:
        lines.append("Strength: No date anomalies detected.")
    contact = analysis_results.get("contact_risks", {})
    if contact.get("email_risk") == "High":
        lines.append("Weakness: Missing or invalid email address.")
    if contact.get("phone_risk") == "High":
        lines.append("Weakness: Missing or invalid phone number.")
    if contact.get("email_risk") == "Low" and contact.get("phone_risk") == "Low":
        lines.append("Strength: Valid contact details provided.")
    ai_risk = analysis_results.get("ai_risk", 0)
    if ai_risk > 50:
        lines.append(f"Weakness: The resume may appear AI-generated (score: {ai_risk}%).")
    else:
        lines.append(f"Neutral: AI-generation likelihood is {ai_risk}%.")
    return "\n".join(f"- {l}" for l in lines)

# =====================================================
# 12) Advanced Semantic Search in Resume
# =====================================================
def semantic_search_in_resume(query: str, resume_text: str, top_k=3):
    sentences = nltk.sent_tokenize(resume_text)
    model = models["semantic_search_model"]
    query_emb = model.encode(query, convert_to_tensor=True)
    sent_embs = model.encode(sentences, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_emb, sent_embs)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(sentences)))
    results = []
    for idx in range(len(top_results[0])):
        score = float(top_results[0][idx])
        sent_idx = int(top_results[1][idx])
        results.append({
            "sentence": sentences[sent_idx],
            "similarity": round(score, 3)
        })
    return results

# =====================================================
# 13) Skills Matrix, Decision Points, and Skill Gap Analysis
# =====================================================
def extract_years_experience(resume_text, skill):
    text = resume_text.lower()
    skill = skill.lower()
    lines = text.split('\n')
    years = []
    pattern = re.compile(r'(\d+)\s*(\+)?\s*years?')
    for line in lines:
        if skill in line:
            matches = pattern.findall(line)
            for match in matches:
                try:
                    years.append(int(match[0]))
                except:
                    pass
    if years:
        return max(years)
    else:
        return "N/A"

def generate_skills_matrix(resume_text, candidate_answers, role):
    skills = custom_criteria[role]["required_skills"]
    rows = []
    for skill in skills:
        present_in_resume = "Yes" if skill.lower() in resume_text.lower() else "No"
        if candidate_answers and candidate_answers.strip():
            present_in_answers = "Yes" if skill.lower() in candidate_answers.lower() else "No"
        else:
            present_in_answers = "N/A"
        years_exp = extract_years_experience(resume_text, skill)
        gap_decision = "Missing" if present_in_resume == "No" and present_in_answers == "No" else "Sufficient"
        rows.append({
            "Required Skill": skill,
            "Found in Resume": present_in_resume,
            "Found in Answers": present_in_answers,
            "Years of Experience": years_exp,
            "Gap Decision": gap_decision
        })
    return pd.DataFrame(rows)

def generate_decision_points(resume_text, candidate_answers, role):
    criteria = custom_criteria[role]
    missing_skills = []
    for skill in criteria["required_skills"]:
        if skill.lower() not in resume_text.lower() and (not candidate_answers or skill.lower() not in candidate_answers.lower()):
            missing_skills.append(skill)
    missing_education = []
    for edu in criteria["education"]:
        if edu.lower() not in resume_text.lower() and (not candidate_answers or edu.lower() not in candidate_answers.lower()):
            missing_education.append(edu)
    missing_experience = []
    for exp in criteria["experience"]:
        if exp.lower() not in resume_text.lower() and (not candidate_answers or exp.lower() not in candidate_answers.lower()):
            missing_experience.append(exp)
    return missing_skills, missing_education, missing_experience

def generate_skill_gap_analysis(resume_text, candidate_answers, role):
    missing_skills, missing_edu, missing_exp = generate_decision_points(resume_text, candidate_answers, role)
    report_lines = []
    if missing_skills:
        report_lines.append("Missing Skills: " + ", ".join(missing_skills))
    else:
        report_lines.append("All required skills are present.")
    if missing_edu:
        report_lines.append("Missing Education Topics: " + ", ".join(missing_edu))
    else:
        report_lines.append("No major education gaps identified.")
    if missing_exp:
        report_lines.append("Missing Experience Areas: " + ", ".join(missing_exp))
    else:
        report_lines.append("Experience areas are well covered.")
    return "\n".join(report_lines)

# =====================================================
# 14) LinkedIn Profile Analysis Functions
# =====================================================
def get_linkedin_profile_text(url):
    """
    Attempts to retrieve the LinkedIn profile text from the URL.
    For demonstration, if the URL contains 'kemal-e-6a2909123',
    return a dummy profile text.
    """
    if "kemal-e-6a2909123" in url:
        return (
            "Kemal E is an experienced data scientist with over 8 years of experience in machine learning, "
            "data analysis, and statistical modeling. He has worked at leading tech companies and has a strong "
            "background in Python, R, and SQL. His expertise includes deep learning, predictive modeling, and NLP. "
            "Kemal holds a Master's degree in Computer Science and has published several research papers."
        )
    else:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                profile_text = soup.get_text(separator="\n")
                return profile_text
            else:
                return ""
        except Exception as e:
            return ""

def analyze_linkedin_profile(linkedin_text, jd_text, role):
    results = {}
    results["plagiarism_score"] = detect_copied_content(linkedin_text, jd_text)["plagiarism_score"]
    results["skill_match"] = enhanced_skill_analysis(linkedin_text, jd_text, role)
    return results

def evaluate_linkedin_quality(linkedin_text, jd_text, role):
    analysis = analyze_linkedin_profile(linkedin_text, jd_text, role)
    quality_score = 100.0
    plagiarism = analysis.get("plagiarism_score", 0)
    if plagiarism > 30:
        quality_score -= (plagiarism - 30) * 0.5
    sm = analysis.get("skill_match", {})
    avg_skill = (sm.get("technical", 0) + sm.get("domain", 0) + sm.get("experience", 0) + sm.get("education", 0)) / 4
    if avg_skill < 50:
        quality_score -= (50 - avg_skill) * 0.5
    else:
        quality_score += (avg_skill - 50) * 0.2
    missing_skills, missing_edu, missing_exp = generate_decision_points(linkedin_text, "", role)
    total_missing = len(missing_skills) + len(missing_edu) + len(missing_exp)
    quality_score -= total_missing * 2
    quality_score = max(0, min(quality_score, 100))
    if quality_score >= 80:
        rating = "High"
    elif quality_score >= 50:
        rating = "Medium"
    else:
        rating = "Low"
    if quality_score >= 75:
        decision = "High Confidence to Hire (LinkedIn)"
    elif quality_score >= 50:
        decision = "Moderate Confidence to Hire (LinkedIn)"
    else:
        decision = "Low Confidence to Hire (LinkedIn)"
    summary = f"LinkedIn Quality Score: {quality_score:.1f} ({rating} Quality). Decision: {decision}."
    return quality_score, rating, summary

# =====================================================
# 15) Candidate Quality Evaluation and Decision Points
# =====================================================
def evaluate_candidate_quality(analysis_results, verification_results, resume_text, candidate_answers, role):
    quality_score = 100.0
    plagiarism = analysis_results.get("plagiarism_score", 0)
    if plagiarism > 30:
        quality_score -= (plagiarism - 30) * 0.5
    not_typical_count = sum(1 for item in verification_results if item["verdict"] == "No")
    total_verified = len(verification_results) if verification_results else 1
    not_typical_ratio = not_typical_count / total_verified
    quality_score -= not_typical_ratio * 20
    sm = analysis_results.get("skill_match", {})
    avg_skill = (sm.get("technical", 0) + sm.get("domain", 0) + sm.get("experience", 0) + sm.get("education", 0)) / 4
    if avg_skill < 50:
        quality_score -= (50 - avg_skill) * 0.5
    else:
        quality_score += (avg_skill - 50) * 0.2
    missing_skills, missing_edu, missing_exp = generate_decision_points(resume_text, candidate_answers, role)
    total_missing = len(missing_skills) + len(missing_edu) + len(missing_exp)
    quality_score -= total_missing * 2
    quality_score = max(0, min(quality_score, 100))
    if quality_score >= 80:
        rating = "High"
    elif quality_score >= 50:
        rating = "Medium"
    else:
        rating = "Low"
    if quality_score >= 75:
        decision = "High Confidence to Hire"
    elif quality_score >= 50:
        decision = "Moderate Confidence to Hire"
    else:
        decision = "Low Confidence to Hire"
    summary = f"Candidate Quality Score: {quality_score:.1f} ({rating} Quality). Decision: {decision}."
    return quality_score, rating, summary

# =====================================================
# 16) Main Analysis Pipeline (Non-GPT Based)
# =====================================================
def analyze_resume(resume_text, jd_text, role):
    results = {}
    results["ai_risk"] = 0
    plag = detect_copied_content(resume_text, jd_text)
    results.update(plag)
    dts, anoms = detect_date_anomalies(resume_text)
    results["dates"] = dts
    results["date_anomalies"] = anoms
    contact_ = detect_contact_inconsistencies(resume_text)
    results["contact_risks"] = contact_
    sm = enhanced_skill_analysis(resume_text, jd_text, role)
    results["skill_match"] = sm
    return results

# =====================================================
# 17) Generate Semantic Search Report for HR
# =====================================================
def generate_semantic_report(query, resume_text, top_k=3):
    results = semantic_search_in_resume(query, resume_text, top_k)
    combined_text = " ".join([item["sentence"] for item in results])
    return combined_text

# =====================================================
# 18) Generate Interview Questions Based on Missing Items
# =====================================================
def generate_interview_questions(resume_text, candidate_answers, role):
    missing_skills, missing_edu, missing_exp = generate_decision_points(resume_text, candidate_answers, role)
    questions = []
    for skill in missing_skills:
        questions.append(f"Can you describe your experience with {skill}?")
    for edu in missing_edu:
        questions.append(f"What education or training have you pursued in {edu}?")
    for exp in missing_exp:
        questions.append(f"Can you elaborate on your experience with {exp}?")
    return questions
def flag_responsibilities_projects(resume_text):
    """
    Extracts sentences from the resume that might describe responsibilities or projects.
    These are typically sentences that start with verbs (indicating actions taken).
    """
    sentences = nltk.sent_tokenize(resume_text)
    flagged = []
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        if words and words[0].lower() in ["developed", "implemented", "led", "managed", "designed", "built",
                                          "engineered", "optimized", "executed", "analyzed", "researched"]:
            flagged.append({"excerpt": sent})
    return flagged
def analyze_flagged_responsibilities(resume_text, role):
    """
    Analyzes flagged responsibilities using a generative model.
    Calls `generative_analysis_responsibility` for each flagged responsibility.
    """
    flagged = flag_responsibilities_projects(resume_text)
    analysis_results = []
    for item in flagged:
        res_text = item["excerpt"]
        result = generative_analysis_responsibility(res_text, role)
        analysis_results.append(result)
    return analysis_results

# =====================================================
# 19) Main Streamlit App
# =====================================================
def main():
    # Do not start analysis until a LinkedIn URL is provided.
    linkedin_url = st.sidebar.text_input("LinkedIn Profile URL (Required to start analysis)", key="linkedin_url")
    if not linkedin_url.strip():
        st.info("Please provide a LinkedIn profile URL to start the analysis.")
        st.stop()
    
    if fallback_spacy_message:
        st.warning(fallback_spacy_message)
    
    st.title("🔍 AI-Powered Resume Fraud Detection (DeBERTa + Semantic Search)")
    st.markdown(
        """
        ### This tool performs:
        - Fraud detection (plagiarism, date anomalies, contact issues)
        - Job responsibility checks via zero-shot classification
        - Skill matching and gap analysis with a skills matrix (including years of experience)
        - Identification of missing skills, education, and experience areas
        - Semantic search within the resume and a generated semantic report for HR
        - Candidate Quality Evaluation to aid HR decision-making
        - LinkedIn Profile Analysis compared to the job description
        - Comparison of resume responsibilities to the job description
        - Generation of interview questions for missing areas
        """
    )
    
    with st.sidebar:
        selected_role = st.selectbox("Select Target Role", list(custom_criteria.keys()))
        jd_input = st.text_area("Paste Job Description", height=200)
        uploaded_resume = st.file_uploader("Upload Candidate Resume", type=["pdf", "docx", "txt"])
        candidate_answers = st.text_area("Candidate Interview Answers (Optional)", height=150)
        semantic_query = st.text_input("Semantic Search Query (Optional)", key="semantic_query")
        top_k = st.number_input("Top K results for Semantic Search", min_value=1, max_value=10, value=3)
    
    col1, col2 = st.columns([2, 1])
    
    analysis_results = {}
    resume_text = ""
    verification_results = []
    
    with col1:
        with st.spinner("Waiting for analysis to begin..."):
            pass
        
        if uploaded_resume:
            resume_text = parse_resume(uploaded_resume)
            if resume_text.strip() and jd_input.strip():
                with st.spinner("Analyzing resume and job description..."):
                    analysis_results = analyze_resume(resume_text, jd_input, selected_role)
                
                st.subheader("🧐 Comprehensive Analysis Report")
                
                with st.spinner("Computing risk indicators..."):
                    risk_factors = {
                        "AI Generation Risk": analysis_results["ai_risk"],
                        "Plagiarism Risk": analysis_results["plagiarism_score"],
                        "Date Anomalies": len(analysis_results["date_anomalies"]),
                        "Contact Risk": 75 if analysis_results["contact_risks"]["email_risk"] == "High" else 25
                    }
                    fig_risk = create_risk_assessment(risk_factors)
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                with st.spinner("Evaluating skill match..."):
                    sm = analysis_results["skill_match"]
                    radar = create_skill_radar({
                        "Technical Skills": sm["technical"],
                        "Domain Knowledge": sm["domain"],
                        "Experience Relevance": sm["experience"],
                        "Education Fit": sm["education"]
                    })
                    st.plotly_chart(radar, use_container_width=True)
                
                with st.spinner("Comparing resume with job description..."):
                    c_pairs = analysis_results["copied_pairs"]
                    if not c_pairs:
                        st.info("No high-overlap lines found.")
                    else:
                        rows = []
                        for (rs, js, ratio) in c_pairs:
                            rows.append({
                                "Resume Snippet": rs,
                                "JD Snippet": js,
                                "Similarity": f"{ratio:.2f}"
                            })
                        st.table(pd.DataFrame(rows))
                
                with st.spinner("Comparing resume responsibilities to job description..."):
                    comp_results = compare_resume_responsibilities(resume_text, jd_input, selected_role)
                    if comp_results:
                        st.subheader("Responsibilities Comparison with Job Description")
                        st.table(pd.DataFrame(comp_results))
                    else:
                        st.info("No flagged responsibilities found for comparison.")
                
                with st.spinner("Running generative analysis on flagged responsibilities..."):
                    gen_results = analyze_flagged_responsibilities(resume_text, selected_role)
                    if gen_results:
                        df_gen = pd.DataFrame(gen_results)
                        st.table(df_gen[["responsibility", "verdict", "reasoning", "confidence"]])
                        conf_values = []
                        for r in gen_results:
                            try:
                                conf = float(r["confidence"].replace("%", ""))
                                conf_values.append(conf)
                            except:
                                pass
                        if conf_values:
                            avg_conf = np.mean(conf_values)
                            st.markdown(f"**Total Confidence Level: {avg_conf:.1f}%**")
                        else:
                            st.markdown("**Total Confidence Level: N/A**")
                    else:
                        st.info("No flagged responsibilities for generative analysis.")
                
                with st.spinner("Generating detailed strength & weakness analysis..."):
                    detail_report = generate_detailed_analysis(analysis_results, selected_role)
                    st.write(detail_report)
                
                with st.spinner("Computing skills matrix and gap analysis..."):
                    st.subheader("Skills Matrix")
                    skills_matrix = generate_skills_matrix(resume_text, candidate_answers, selected_role)
                    st.table(skills_matrix)
                    st.subheader("Skill Gap Analysis Report")
                    gap_report = generate_skill_gap_analysis(resume_text, candidate_answers, selected_role)
                    st.write(gap_report)
                    missing_skills, missing_edu, missing_exp = generate_decision_points(resume_text, candidate_answers, selected_role)
                    st.subheader("Decision Points")
                    st.markdown("**Missing Skills:** " + (", ".join(missing_skills) if missing_skills else "None"))
                    st.markdown("**Missing Education Topics:** " + (", ".join(missing_edu) if missing_edu else "None"))
                    st.markdown("**Missing Experience Areas:** " + (", ".join(missing_exp) if missing_exp else "None"))
                
                with st.spinner("Evaluating overall candidate quality..."):
                    quality_score, quality_rating, quality_summary = evaluate_candidate_quality(
                        analysis_results, verification_results, resume_text, candidate_answers, selected_role
                    )
                    st.write(quality_summary)
                
                with st.spinner("Generating semantic search report for HR..."):
                    default_query = "Summarize the candidate's key strengths and weaknesses."
                    if not semantic_query.strip():
                        semantic_query = default_query
                        st.info(f"No query provided. Using default query: '{default_query}'")
                    semantic_report = generate_semantic_report(semantic_query, resume_text, top_k=top_k)
                    st.markdown("**Semantic Search Report:**")
                    st.write(semantic_report)
                
                with st.spinner("Generating interview questions based on missing areas..."):
                    interview_questions = generate_interview_questions(resume_text, candidate_answers, selected_role)
                    if interview_questions:
                        st.subheader("Interview Questions for Candidate")
                        for q in interview_questions:
                            st.markdown(f"- {q}")
                    else:
                        st.info("No interview questions generated. The resume and JD seem to cover all areas.")
                
                with st.spinner("Analyzing LinkedIn profile..."):
                    linkedin_text = get_linkedin_profile_text(linkedin_url)
                    if linkedin_text:
                        st.subheader("🔗 LinkedIn Profile Analysis Report")
                        linkedin_analysis = analyze_linkedin_profile(linkedin_text, jd_input, selected_role)
                        st.markdown("**LinkedIn Plagiarism Score:** " + str(linkedin_analysis.get("plagiarism_score", "N/A")))
                        st.markdown("**LinkedIn Skill Match:**")
                        linkedin_sm = linkedin_analysis.get("skill_match", {})
                        st.write(linkedin_sm)
                        st.subheader("LinkedIn Skills Matrix")
                        linkedin_matrix = generate_skills_matrix(linkedin_text, "", selected_role)
                        st.table(linkedin_matrix)
                        st.subheader("LinkedIn Skill Gap Analysis")
                        linkedin_gap = generate_skill_gap_analysis(linkedin_text, "", selected_role)
                        st.write(linkedin_gap)
                        quality_score_li, rating_li, summary_li = evaluate_linkedin_quality(linkedin_text, jd_input, selected_role)
                        st.subheader("LinkedIn Quality Evaluation")
                        st.write(summary_li)
    
    with col2:
        st.subheader("🛡️ Fraud Indicators")
        if analysis_results:
            ai_val = analysis_results["ai_risk"]
            st.info(f"AI Generation Probability (Demo): {ai_val}%")
            pl_score = analysis_results["plagiarism_score"]
            if pl_score > 40:
                st.error(f"⚠️ High Overlap with JD: {pl_score:.2f}%")
            else:
                st.info(f"Plagiarism Score: {pl_score:.2f}%")
            d_anoms = analysis_results["date_anomalies"]
            if d_anoms:
                st.warning(f"⏳ Date Anomalies: {len(d_anoms)}")
                for da in d_anoms:
                    st.write(f"- {da}")
            c_risk = analysis_results["contact_risks"]
            if c_risk["email_risk"] == "High":
                st.warning("⚠️ No valid email found.")
            if c_risk["phone_risk"] == "High":
                st.warning("⚠️ No valid phone number found.")
        else:
            st.info("Please upload a resume and provide a job description.")
        
        st.subheader("🔎 Semantic Search in Resume")
        if resume_text:
            search_results = semantic_search_in_resume(semantic_query, resume_text, top_k=top_k)
            if not search_results:
                st.info("No search results found.")
            else:
                st.write(f"**Top {len(search_results)} results for:** {semantic_query}")
                st.table(pd.DataFrame(search_results))
    
if __name__ == "__main__":
    if "linkedin_url" not in st.session_state:
        st.session_state["linkedin_url"] = ""
    if "semantic_query" not in st.session_state:
        st.session_state["semantic_query"] = ""
    main()
