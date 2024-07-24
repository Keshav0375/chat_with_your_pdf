import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from main import get_conversation_chain
from HTMLTemplates import css, bot_template, user_template
import requests
load_dotenv()
OPENAI_API_TYPE = os.environ['OPENAI_API_TYPE']
OPENAI_API_BASE = os.environ['OPENAI_API_BASE']
OPENAI_API_VERSION = os.environ['OPENAI_API_VERSION']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
DEPLOYMENT_NAME = os.environ['TEXT_DEPLOYMENT_NAME']


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunk(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectors(chunks):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=DEPLOYMENT_NAME,
        openai_api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_BASE
    )
    vector_stores = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_stores


def process_pdfs(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunk(raw_text)
    vector_stores = get_vectors(text_chunks)
    vector_stores.save_local('faiss_index')
    st.session_state.vector_stores = vector_stores
    return "Processed Successfully"


def handle_userinput(user_question):
    if not st.session_state.chat_history:
        st.session_state.chat_history = []
    response = requests.post("http://127.0.0.1:8000/ask", json={"question": user_question, "chat_history": st.session_state.chat_history})
    if response.status_code == 200:
        result = response.json()
        st.session_state.chat_history = result["chat_history"]
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    else:
        st.error("Failed to get a response from the API")


def main():
    st.set_page_config(page_title="Conversational PDF Chatbot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vector_stores" not in st.session_state:
        st.session_state.vector_stores = None
    st.header("Chat with Multiple PDFS :books:")
    user_question = st.text_input("Ask a question about your document!")
    if user_question:
        handle_userinput(user_question)
    st.write(user_template.replace("{{MSG}}", "Hello Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your Documents :books:")
        pdf_docs = st.file_uploader("Upload Your PDF here and click process", type="pdf", accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("Processing...."):
                process_pdfs(pdf_docs)
                st.success("Processed Successfully")
                requests.post("http://127.0.0.1:8000/refresh_index")


if __name__ == '__main__':
    main()
