import streamlit as st
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

st.title("📚 Ask Your Book – Improved Version")

# Load models once
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return embed_model, qa_model

embed_model, qa_model = load_models()

# Read PDF safely with optional page limit
def read_pdf(file, max_pages=20):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text += page.get_text()
    return text

# Chunk with overlap
def split_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Build FAISS index
def build_index(chunks):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Find top relevant chunks
def find_relevant_chunks(question, chunks, index, top_k=3):
    q_embed = embed_model.encode([question])
    _, results = index.search(np.array(q_embed), top_k)
    return [chunks[i] for i in results[0]]

# Ask QA model
def answer_question(question, context):
    try:
        result = qa_model(question=question, context=context)
        return result["answer"]
    except:
        return "Sorry, I couldn't find an answer."

# UI
pdf = st.file_uploader("📄 Upload a textbook PDF (max ~10MB recommended)", type="pdf")

if pdf:
    if pdf.size > 10_000_000:
        st.warning("File is too large. Try a smaller PDF (less than 10MB).")
    else:
        with st.spinner("Reading and processing PDF..."):
            text = read_pdf(pdf, max_pages=30)
            chunks = split_text(text)
            index, _ = build_index(chunks)
        st.success("PDF processed! Now ask your question.")

        question = st.text_input("❓ Type your question")

        if question and st.button("Get Answer"):
            relevant_chunks = find_relevant_chunks(question, chunks, index)
            context = " ".join(relevant_chunks)
            answer = answer_question(question, context)
            st.markdown(f"**Answer:** {answer}")
