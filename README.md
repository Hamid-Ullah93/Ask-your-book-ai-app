# 📘 Ask Your Book – AI Study Assistant

**Ask Your Book** is an AI-powered web app where students can upload a textbook chapter (PDF) and ask questions. The app gives answers using Natural Language Processing and Hugging Face models.

---

## 🚀 Features

- Upload textbook PDF
- Ask any question in English
- AI finds answers from the book content
- Uses sentence embeddings and vector search

---

## 🧠 Tech Stack

- **Streamlit** – Web interface
- **Hugging Face Transformers** – Question Answering model
- **Sentence Transformers** – For embeddings
- **FAISS** – For similarity search
- **PyMuPDF** – To extract text from PDF

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
