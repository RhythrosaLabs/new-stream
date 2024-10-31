# document_processor.py

import os
import tempfile
import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def process_uploaded_file(uploaded_file, openai_api_key):
    """
    Process the uploaded file:
    1. Save to a temporary file.
    2. Load and split the document.
    3. Create embeddings and vector store.
    """
    if not uploaded_file:
        return None, "No document uploaded."

    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Choose the appropriate loader based on file type
        if uploaded_file.name.lower().endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = UnstructuredFileLoader(tmp_file_path)

        documents = loader.load()

        # Split documents and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Clean up temporary file
        os.unlink(tmp_file_path)

        return vectorstore, "Document uploaded and processed successfully."
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None, f"Error processing document: {e}"
