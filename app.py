# --- Imports ---
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import torch
from transformers import pipeline
st.set_page_config(page_title="Study mate", page_icon="📚")

# --- Cache the QA Model ---
@st.cache_resource
def load_qa_model():
    """Loads a Hugging Face Question Answering model only once."""
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

# --- Helper Functions ---
def split_text(text, max_words=200):
    """Splits text into chunks with a maximum word limit."""
    words = text.split()
    return [
        " ".join(words[i:i+max_words])
        for i in range(0, len(words), max_words)
    ]

def find_relevant_chunk(question, chunks):
    """Finds the most relevant chunk based on question overlap."""
    best_chunk = None
    best_score = 0
    for chunk in chunks:
        overlap = len(set(question.lower().split()) & set(chunk.lower().split()))
        if overlap > best_score:
            best_score = overlap
            best_chunk = chunk
    return best_chunk

# --- Title & Description ---
st.title("📚 Study Mate – Your AI VTU Syllabus Assistant")
st.subheader("Upload your syllabus PDF, ask a question, and get accurate answers instantly!")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
chunk_size = st.sidebar.slider("Chunk size (words)", min_value=100, max_value=500, value=200, step=50)
show_confidence = st.sidebar.checkbox("Show confidence score", value=True)

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "📄 Upload Your VTU Syllabus PDF",
    type="pdf",
    help="Make sure it's text-based for best results"
)

pdf_text = ""
if uploaded_file:
    with st.spinner("📄 Reading PDF..."):
        pdf = PdfReader(uploaded_file)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text + "\n"

    with st.expander("🔍 View Extracted Text", expanded=False):
        st.text_area("Extracted Text", pdf_text, height=300)

# --- Question Input ---
user_question = st.text_input("💡 Ask your syllabus question:")

if st.button("🔎 Get Answer"):
    if not user_question.strip() or not pdf_text.strip():
        st.warning("⚠️ Please provide both a PDF and a question!")
    else:
        qa = load_qa_model()
        chunks = split_text(pdf_text, max_words=chunk_size)
        relevant_chunk = find_relevant_chunk(user_question, chunks)

        if relevant_chunk:
            with st.spinner("🤖 Thinking..."):
                result = qa(question=user_question, context=relevant_chunk)

            st.success(f"**Answer:** {result['answer']}")
            if show_confidence:
                st.info(f"Confidence Score: {result['score']:.3f}")
        else:
            st.error("❌ No relevant text found in the PDF.")

