import streamlit as st
import spacy
import pdfplumber
import fitz  # pip install pymupdf
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Extract text from uploaded PDF (with fallback)
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""

    # Try pdfplumber first
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.warning(f"pdfplumber failed: {e}")

    # Fallback to PyMuPDF if text is empty
    if not text.strip():
        try:
            uploaded_file.seek(0)  # ← Reset file pointer
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text() + "\n"
        except Exception as e:
            st.error(f"PyMuPDF also failed: {e}")

    return text


# -----------------------------
# Preprocess text
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


# -----------------------------
# Extract keywords
# -----------------------------
def extract_keywords(text):
    doc = nlp(text)
    keywords = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                keywords.append(token.lemma_.lower())
    return list(set(keywords))


# -----------------------------
# Similarity Calculation
# -----------------------------
def calculate_similarity(resume, jd):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, jd])
    similarity = cosine_similarity(vectors)[0][1]
    return round(similarity * 100, 2)


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("ATS Resume Screening System")
st.write("Upload Resume and Paste Job Description")

# Upload Resume
uploaded_resume = st.file_uploader("Upload Resume PDF", type=["pdf"])

# Enter Job Description
job_description = st.text_area("Paste Job Description")

# Analyze Button
if st.button("Analyze Resume"):

    if uploaded_resume is not None and job_description != "":

        # Extract resume text
        resume_text = extract_text_from_pdf(uploaded_resume)

        # ── DEBUG BLOCK ──────────────────────────────────────
        st.write("📄 Resume character count:", len(resume_text))
        st.write("📄 Resume preview:", resume_text[:500] if resume_text else "⚠️ EMPTY")

        if not resume_text.strip():
            st.error("Could not extract text from PDF. Try a different PDF (non-scanned).")
            st.stop()
        # ─────────────────────────────────────────────────────

        st.write("Resume Text:")
        st.write(resume_text[:1000])

        # Preprocess
        resume_clean = preprocess_text(resume_text)
        jd_clean = preprocess_text(job_description)

        # Extract keywords
        resume_keywords = extract_keywords(resume_clean)
        jd_keywords = extract_keywords(jd_clean)

        # Match keywords
        matched_keywords = set(resume_keywords).intersection(set(jd_keywords))
        missing_keywords = set(jd_keywords) - set(resume_keywords)

        # Similarity Score
        score = calculate_similarity(resume_clean, jd_clean)

        # Display Results
        st.subheader("ATS Result")
        st.write(f"Match Score: {score}%")
        st.write("Matched Keywords:")
        st.write(matched_keywords)
        st.write("Missing Keywords:")
        st.write(missing_keywords)

    else:
        st.warning("Please upload resume and enter job description")